import asyncio
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from html import escape
import string
from typing import Any
from urllib.parse import quote_plus, urlparse, urlunparse
import uuid
from pyppeteer import launch, connect
from loguru import logger
import requests
import re
import argparse
import difflib
import urllib3

urllib3.disable_warnings()


PARAM_TEMPLATE_FORMENCODED = ("", "%k=%v", "&", "")
PARAM_TEMPLATE_JSON = ("{", '"%k":"%v"', ",", "}")
PARAM_TEMPLATE_XML = (
    '<?xml version="1.0" encoding="UTF-8"?><root>\n',
    " <%k>%v</%k>",
    "\n",
    "\n</root>",
)


class Handlers:
    def __init__(self, new_headers: dict = {}):
        self.new_headers = {k.lower(): v for k, v in new_headers.items()}

    def request_handler(self, request):
        # if request.url.endswith('.png') or request.url.endswith('.jpg'):
        #     await request.abort()
        asyncio.create_task(request.continue_())

    def console_handler(self, event):
        if event.type == "error":
            print(event.type, event.text, event.args)


@dataclass
class BaseRequest:
    base_request: requests.Response
    base_reflexions: tuple[int, int]


class Comparer:
    base_request: requests.Response

    def __init__(self, base_request: BaseRequest):
        self.base_request = base_request.base_request
        self.base_reflexions = base_request.base_reflexions

    def _compare_statuses(self, current_request: requests.Response):
        return self.base_request.status_code == current_request.status_code, f"{self.base_request.status_code}:{current_request.status_code}"

    def _compare_texts(
        self,
        current_request: requests.Response,
        params_string: str,
        threshold: float = 0.98,
    ):
        base_text = self.base_request.text
        current_text = current_request.text.replace(params_string, "")
        diff = difflib.SequenceMatcher(None, base_text, current_text).ratio()
        if diff >= threshold:
            return True, diff
        return False, diff

    def _compare_times(
        self, current_request: requests.Response, tolerance: float = 1.3
    ):
        high = current_request.elapsed.seconds + tolerance
        low = current_request.elapsed.seconds - tolerance
        if low < self.base_request.elapsed.seconds < high:
            return True, f"l:{low} < base:{self.base_request.elapsed.seconds} < h:{high}"
        return False, f"l:{low} < base:{self.base_request.elapsed.seconds} < h:{high}"

    def _compare_reflexions(self, current_request: requests.Response, raw_params: dict):
        for p, v in raw_params.items():
            if current_request.text.count(v) != self.base_reflexions[1]:
                return False, p
        return True, False

    def is_equal(
        self,
        current_request: requests.Response,
        formatted_params_string: str,
        raw_params: dict,
    ) -> bool:
        status_comparison,_ = self._compare_statuses(current_request)
        times_comparison,_ = self._compare_times(current_request)
        texts_comparison,_ = self._compare_texts(current_request, formatted_params_string)
        reflexions_comparison, _ = self._compare_reflexions(current_request, raw_params)

        # print(status_comparison, times_comparison, texts_comparison, reflexions_comparison)
        return (
            status_comparison
            and times_comparison
            and texts_comparison
            and reflexions_comparison
        )


class Web_classic:

    base_headers: dict = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36"
    }

    def __init__(
        self, forced_headers: dict = {}, timeout: int = 30, method: str = "GET"
    ):
        self.page = None
        self.results = dict()
        self.forced_headers = forced_headers
        self.timeout = timeout
        self.load_headers()
        self.start_browser()

    def load_headers(self):
        self.headers = {}
        self.headers.update(self.base_headers)
        self.headers.update(self.forced_headers)

    def start_browser(
        self,
    ):
        self.browser = requests.Session()

    def page_goto(
        self, url: str, data: str = "", method: str = "GET"
    ) -> None | requests.Response:
        try:
            self.page = getattr(self.browser, method.lower())(
                url, timeout=self.timeout, headers=self.headers, verify=False, data=data
            )
        except Exception as e:
            logger.error(f"Error: {e}")

            if not url.startswith("https") and url.startswith("http"):
                logger.info(f"Trying https for {url}")
                return self.page_goto(f"https{url[4:]}", data=data, method=method)
            elif url.startswith("https"):
                pass
            else:
                logger.critical(f'Url {url} does not start at least with "http" :/')
            return None

        return self.page

    def get_page_content(self):
        self.page: requests.Response
        try:
            return self.page.text
        except AttributeError:
            logger.error(f"Error :/ request failed ?")
        return ""


class Web_headless:
    pages: dict

    def __init__(
        self,
        forced_headers: dict = {},
        timeout: int = 30,
        attach_to_existing_chrome: bool = False,
        open_new_tabs: bool = False,
        pyppeteer_args: dict = {
            "headless": True,
            "ignoreHTTPSErrors": True,
            "handleSIGTERM": False,
            "handleSIGINT": False,
            "executablePath": "/usr/bin/chromium-browser",
            "devtools": False,
            "args": ["--no-sandbox"],
        },
    ):

        self.pyppeteer_args = pyppeteer_args
        self.pages = list()
        self.results = dict()
        self.forced_headers = forced_headers
        self.timeout = timeout
        self.attach_to_existing_chrome = attach_to_existing_chrome
        self.open_new_tabs = open_new_tabs
        self.Handlers = Handlers(new_headers=forced_headers)

    async def start_browser(
        self,
    ):
        if self.attach_to_existing_chrome:
            logger.debug(
                f"Attaching to chrome on port {self.attach_to_existing_chrome}"
            )
            self.browser = await connect(
                browserURL=f"http://127.0.0.1:{self.attach_to_existing_chrome}"
            )
        else:
            self.browser = await launch(self.pyppeteer_args)

    async def new_page(self):
        if self.open_new_tabs:
            logger.debug("Spawning new page for link")
            self.pages.append(await self.browser.newPage())
            id_ = len(self.pages) - 1
            await self.pages[id_].setViewport({"defaultViewport": None})
            # set request handler to override headers when request
            await self.pages[id_].setRequestInterception(True)
            self.pages[id_].on("request", self.Handlers.request_handler)
            self.pages[id_].on("console", self.Handlers.console_handler)

            self.pages[id_].setDefaultNavigationTimeout(self.timeout * 1000)

        else:
            pages = await self.browser.pages()

            if len(pages) == 0:
                logger.debug("There is no current page, Spawning new page for link")
                self.pages.append(await self.browser.newPage())
                await self.pages[id_].setViewport({"defaultViewport": None})
                # set request handler to override headers when request
                await self.pages[id_].setRequestInterception(True)
                self.pages[id_].on("request", self.Handlers.request_handler)
                self.pages[id_].on("console", self.Handlers.console_handler)
                self.pages[id_].setDefaultNavigationTimeout(self.timeout * 1000)
                self.results.setdefault(id_, dict())
            else:
                self.pages = pages[:1]
            id_ = 0
        self.results.setdefault(id_, dict())

        return id_

    async def page_goto(self, id, url, headers={}):
        try:
            h = self.forced_headers.copy()
            h.update(headers)
            await self.pages[id].setExtraHTTPHeaders(h)
            p = await self.pages[id].goto(url, waitUntil="networkidle0")
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
        self.results[id]["url"] = url
        return p

    async def get_page_content(self, id):
        try:
            body = await self.pages[id].content()
            self.results[id]["body"] = body
            return body
        except Exception as e:
            logger.error(f"Error: {e}")
            self.results[id]["body"] = ""
            return ""

    async def close_browser(self):
        await self.browser.close()

    async def screenshot_page(self, id, path):
        await self.pages[id].screenshot({"path": f"/tmp/{path}.png"})


class ParamFormat:

    paramformat: str
    header: str
    footer: str
    separator: str

    def __init__(self, format):
        
        self.header = format[0]
        self.paramformat = format[1]
        self.separator = format[2]
        self.footer = format[3]
        logger.debug(f"Using special format: {self.__str__()}")

    def generate_string(self, key, value, incl_sep: bool = True):
        return f"{self.paramformat.replace('%k', key).replace('%v', value)}{self.separator if incl_sep else ''}"

    def get_min_length(self):
        p = self.generate_string("", "", False)
        return len(self.header + p + self.footer)
    
    def __str__(self):
        return f"{self.header}{self.generate_string('foo', 'bar')}{self.generate_string('foo2','bar2',False)}{self.footer}"


class Miner:

    web: Web_classic | Web_headless
    url: str
    data: str
    methods: list
    wordlist_path: str
    wordlist: list
    max_uri_length: int = 8000  # 8kB d'URL par défaut
    base_request: Any
    as_body: bool | None
    threads_number: int
    found_params: dict

    def __init__(
        self,
        web: Web_classic | Web_headless,
        url: str,
        data: str,
        methods: list = ["GET"],
        wordlist: str = "",
        threads_number: int = 3,
        params_format: str | None = None,
        as_body: bool | None = False,
    ) -> None:
        self.web = web
        self.url = url
        self.methods = methods
        self.wordlist_path = wordlist
        self.data = data
        self.parameter_template = self.select_param_format(params_format)
        self.as_body = as_body
        self.threads_number = threads_number

    @staticmethod
    def as_body_selector(value: bool | None, method: str) -> bool:
        match method:
            case "GET":
                if value is None or not value:
                    return False
                if value:
                    return True
            case _:
                if value:
                    return True
        return False

    def select_param_format(self, params_format: str | None):
        match params_format:
            case None | "formencoded":
                return PARAM_TEMPLATE_FORMENCODED
            case "json":
                return PARAM_TEMPLATE_JSON
            case "xml":
                return PARAM_TEMPLATE_XML
            case _:
                return params_format.split("§")

    @staticmethod
    def gen_random_value(size: int = 5):
        return uuid.uuid4().__str__()[:size]

    def load_wordlist(self):
        if not self.wordlist_path:
            exit("No wordlist set")
        try:
            with open(self.wordlist_path) as wd:
                self.wordlist = wd.read().splitlines()
        except IOError as e:
            logger.error(f"Wordlist not found: {e}")

    @staticmethod
    def gen_format_params(
        params: dict, maxlen: int, pf: ParamFormat, pack_needed: int = -1
    ) -> list[tuple[str, dict]]:
        formatted_list = []
        formatted = ""
        raw_pack = {}
        formatted += pf.header

        for i, (param, value) in enumerate(params.items()):
            param_batch = pf.generate_string(param, value)
            if len(formatted + param_batch + pf.footer) <= maxlen:
                if i == len(params.keys()) - 1:
                    formatted += param_batch[: -(len(pf.separator))] + pf.footer
                    raw_pack.update({param: value})
                    formatted_list.append((formatted, raw_pack))
                else:
                    formatted += param_batch
                    raw_pack.update({param: value})
            else:
                formatted = formatted[: -(len(pf.separator))]
                formatted += pf.footer
                formatted_list.append((formatted, raw_pack))
                if len(formatted_list) == pack_needed:
                    break
                # reset format
                formatted = "" + pf.header + param_batch
                raw_pack = {}
                raw_pack.update({param: value})

        return formatted_list

    @staticmethod
    def merge_parameters(
        paramFormat: ParamFormat, existing_params: str, new_params: str
    ) -> str:
        if existing_params is None or len(existing_params) == 0:
            return new_params
        new_params = re.sub(f"^{paramFormat.header}", "", new_params)
        existing_params = re.sub(f"{paramFormat.footer}$", "", existing_params)
        return existing_params + paramFormat.separator + new_params

    def format_request(
        self,
        url: str,
        data: str,
        method: str,
        formatted_parameters: str,
        param_format: ParamFormat,
        as_body: bool = True,
    ):
        url_parsed = urlparse(url)
        existing_params = url_parsed.query
        # print(formatted_parameters)
        if not as_body:
            new_params = self.merge_parameters(
                param_format, existing_params, formatted_parameters
            )
            url = urlunparse(url_parsed._replace(query=new_params))
        else:
            if not data:
                data = str()
            data = self.merge_parameters(param_format, data, formatted_parameters)

        return url, data, method

    def find_max_length(self, method: str, paramformat: ParamFormat):

        param_list = {
            self.gen_random_value(6): self.gen_random_value(6)
            for _ in range(int(self.max_uri_length / 8))
        }
        current_request: Any | requests.Response
        current_request = None
        size = self.max_uri_length - (
            len(self.url) if method.upper() == "GET" else len(self.data)
        )
        while (
            current_request == None
            or re.findall("41(3|4) ERROR", current_request.text, re.IGNORECASE)
            or any(
                [
                    current_request.status_code == 414,
                    current_request.status_code == 413,
                    current_request.status_code
                    != self.base_request.base_request.status_code,
                ]
            )
        ):
            # reduce size length for each row, give extra space for base row
            size -= 300
            logger.debug(f"Current pack size: {size}")
            pack = self.gen_format_params(param_list, size, paramformat, 1)[0][0]
            sending_url, sending_data, method = self.format_request(
                self.url,
                self.data,
                method,
                pack,
                paramformat,
                as_body=self.as_body_selector(self.as_body, method),
            )
            current_request = self.web.page_goto(sending_url, sending_data, method)
            
        return size - 1000

    def threading(
        self,
        url: str,
        data: str,
        method: str,
        paramFormat: ParamFormat,
        fuzzing_params: str,
        raw_pack: dict,
        original: bool
    ):
        sending_url, sending_data, method = self.format_request(
            url,
            data,
            method,
            fuzzing_params,
            paramFormat,
            as_body=self.as_body_selector(self.as_body, method),
        )
        return (
            fuzzing_params,
            raw_pack,
            self.web.page_goto(sending_url, sending_data, method),
            original
        )

    def find_baseinfo(
        self, method: str, paramFormat: ParamFormat
    ) -> tuple[Any, tuple[int, int]]:
        base_request = self.web.page_goto(self.url, self.data, method)
        param_name = self.gen_random_value(6)
        param_value = self.gen_random_value(6)
        formatted_param = self.gen_format_params(
            {param_name: param_value}, 10000, paramFormat
        )[0][0]
        sending_url, sending_data, method = self.format_request(
            self.url,
            self.data,
            method,
            formatted_param,
            paramFormat,
            as_body=self.as_body_selector(self.as_body, method),
        )
        request_reflects = self.web.page_goto(sending_url, sending_data, method)
        reflexions_value = request_reflects.text.count(param_value)
        reflexions_name = request_reflects.text.count(param_name)

        return (request_reflects, (reflexions_name, reflexions_value))
    
    def start_heuristics_checks(self, method: str, base_request: requests.Response) -> set:
        parameters = set()
        html_general_inputs = re.findall(r'<input.+?name=["\']?([^>"\'\s]+)', base_request.text, flags=re.IGNORECASE | re.DOTALL)
        # forms = re.findall(r"<form.+?[^>]>(.+?)<\/form", base_request.text, flags=re.IGNORECASE | re.DOTALL)
        scripts = re.findall(r"<script.+?[^>]>(.+?)<\/script", base_request.text, flags=re.IGNORECASE | re.DOTALL)
        for script in scripts:
            #checking empty vars in javascript
            #this often results in free xss in the wild

            # find params first way
            for param in re.findall(r'([^\s!=<>]+)\s*=\s*[\'"`][\'"`]', base_request.text, re.IGNORECASE):
                # check if its garbage
                if isText(param):
                    parameters.add(param)
            # find params second way
            for param in re.findall(r'([^\'"]+)[\'"]:\s?[\'"]', base_request.text, re.IGNORECASE):
                if isText(param):
                    parameters.add(param)
        logger.warning(f"Potential parameters found (heuristics): {parameters}")
        return parameters
        

    def mine(self) -> dict:
        self.load_wordlist()
        paramFormat = ParamFormat(self.parameter_template)
        self.found_params = {}
        for method in self.methods:
            self.found_params[method] = set()
            base_request, number_reflexions = self.find_baseinfo(method, paramFormat)

            self.base_request = BaseRequest(base_request, number_reflexions)
            max_length = self.find_max_length(method, paramFormat)
            self.found_params[method].union(self.start_heuristics_checks(method, base_request))
            logger.debug(f"Found max length of {max_length}")

            comparer = Comparer(self.base_request)

            futures = set()
            executor = ThreadPoolExecutor(max_workers=self.threads_number)

            packs = {wd: self.gen_random_value(5) for wd in self.wordlist}
            formatted_packs = self.gen_format_params(packs, max_length, paramFormat)
            futures.update(
                {
                    executor.submit(
                        self.threading,
                        self.url,
                        self.data,
                        method,
                        paramFormat,
                        formatted_pack_,
                        pack_,
                        True
                    )
                    for formatted_pack_, pack_ in formatted_packs
                }
            )
            request_count = 0
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                # for each future done
                for future in done:
                    request_count += 1
                    print(f"Processing request {request_count}, {len(futures)} remains", end="\r") 
                    fuzzing_params: str
                    origin_pack: dict
                    current_request: requests.Response
                    original_request: bool

                    fuzzing_params, origin_pack, current_request, original_request = future.result()
                    if len(origin_pack) < 1:
                        continue
                    if not comparer.is_equal(
                        current_request, fuzzing_params, origin_pack
                    ):
                        # logger.success(f"Pack is different")
                        while not comparer._compare_reflexions(
                            current_request, origin_pack
                        )[0]:
                            p = comparer._compare_reflexions(
                                current_request, origin_pack
                            )[1]
                            logger.warning(f"Differences: Reflects;{p}")
                            del origin_pack[p]
                            self.found_params[method].add(p)
                        statuses, status_reason = comparer._compare_statuses(current_request)
                        texts, text_reason = comparer._compare_texts(current_request, fuzzing_params)
                        times, time_reason = comparer._compare_times(current_request)
                        
                        if (
                            not statuses
                            or not texts
                            or not times
                        ):
                            if original_request:
                                logger.warning(f"Differences:{f' Statuses:{status_reason}' if not statuses else ''}{f' Texts:{text_reason}' if not texts else ''}{f' Times:{time_reason}' if not times else ''}")
                            if len(origin_pack) == 1:
                                self.found_params[method].add(
                                    list(origin_pack.keys())[0]
                                )
                                break
                            # splitting the pack
                            first_half_pack = dict(
                                list(origin_pack.items())[: len(origin_pack) // 2]
                            )
                            # keep the original payload & params
                            second_half_pack = dict(
                                list(origin_pack.items())[len(origin_pack) // 2 :]
                            )
                            # logger.debug("Submitting 2 new packs to process")
                            formatted_pack_1 = self.gen_format_params(
                                first_half_pack, max_length, paramFormat
                            )[0][0]
                            formatted_pack_2 = self.gen_format_params(
                                second_half_pack, max_length, paramFormat
                            )[0][0]
                            futures.update(
                                {
                                    executor.submit(
                                        self.threading,
                                        self.url,
                                        self.data,
                                        method,
                                        paramFormat,
                                        formatted_pack_1,
                                        first_half_pack,
                                        False
                                    )
                                }
                            )
                            futures.update(
                                {
                                    executor.submit(
                                        self.threading,
                                        self.url,
                                        self.data,
                                        method,
                                        paramFormat,
                                        formatted_pack_2,
                                        second_half_pack,
                                        False
                                    )
                                }
                            )

            print(self.found_params)

        return self.found_params


# async def visit_n_parse_headless(self, id_, url):
#     p = await self.web.page_goto(id_, url)
#     try:
#         self.add_visited({url}, status=p.status, headers=p.headers)

#     except:
#         self.add_failed({url})
#         logger.warning(f"STATUS: FAIL | LINK: {url}")

#     self.queue.discard(url)
#     content = await self.web.get_page_content(id_)
#     if p:
#         logger.success(f"STATUS: {p.status} | LENGTH: {len(content)} | LINK: {url}")
#     self.results_pages[url]["len"] = len(content)
#     with open(f"/tmp/crawlalllinks/{id_}.txt", "w") as f:
#         f.write(content)
#     with open(f"/tmp/crawlalllinks/{id_}_decoded.txt", "w") as f:
#         f.write(unescape(unquote(content)))
#     links = self.find_all_links_v3(id_)
#     for link in set(links.splitlines()):
#         retrieved_links.add(link)
#     return set(links.splitlines())


####################


def get_arguments():
    parser = argparse.ArgumentParser(description="Fuzz parameters based on a wordlist")
    parser.add_argument("url", type=str, help="URL to fuzz")
    parser.add_argument("wordlist", type=str, help="File to open and use as a wordlist")
    parser.add_argument(
        "-X",
        "--method",
        action="append",
        help="Add a method to use for fuzzing. Default ['GET']",
        default=None,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Shut the f up and output only the urls",
        default=False,
    )
    parser.add_argument(
        "-mf", "--max-fails", type=int, help="max fails before stopping", default=10
    )
    parser.add_argument("-t", "--threads", type=int, help="max threads", default=3)
    parser.add_argument(
        "-H", "--header", type=str, help="Headers to send", action="append", default=[]
    )
    parser.add_argument(
        "--timeout", type=int, help="Timeout for fetching web page", default=35
    )
    parser.add_argument(
        "-ch",
        "--chrome-headless",
        help="Use a headless browser to run the crawler",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--new-tabs-only",
        action="store_true",
        help="Open new tabs for each payload on chrome (headless or remote debug)",
        default=False,
    )
    parser.add_argument(
        "--chrome-remote-debug",
        help="Specify an existing and openned chrome remote debug port",
        default=None,
    )
    parser.add_argument(
        "-d", "--data", help="Add specific body data", type=str, default=""
    )
    parser.add_argument(
        "--keep-chrome",
        help="Don't close chrome/ium at the end of the crawling process",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-pf",
        "--params-format",
        help="Choose a paramformat (formencoded, json, xml) or create your own (format: header§%%k,%%v§separator§footer)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--as-body",
        help="Send parameters as body (default: auto, only GET method is not as_body)",
        default=None,
        action="store_true",
    )
    return parser.parse_args()


def load_headers(headers_string: list[str]):
    headers = {}
    for header in headers_string:
        header_name, value = header.split(": ")
        headers[header_name] = value
    return headers

def isText(garbage:str) -> bool:
    specials = "_[]-~.âàäçéèêëîïôùûüœ"
    for a in garbage:
        # find if whole param is normal string with few specific caracters
        # TODO: find a way to keep this check but include special foreign caracters (like chineese for instance)
        if not (a.lower() in string.ascii_lowercase or a in string.digits or a in list(specials)+list(quote_plus(specials))+list(escape(specials))):
            return False
    return True


if __name__ == "__main__":
    arguments = get_arguments()
    web = (
        Web_headless(
            load_headers(arguments.header),
            arguments.timeout,
            arguments.chrome_remote_debug,
            arguments.new_tabs_only,
        )
        if arguments.chrome_headless
        else Web_classic(load_headers(arguments.header), arguments.timeout)
    )
    miner = Miner(
        web,
        arguments.url,
        arguments.data,
        arguments.method or ["GET"],
        arguments.wordlist,
        arguments.threads,
        arguments.params_format,
        arguments.as_body,
    )
    miner.mine()


# veb = Web_headless(load_headers(args.header), timeout=args.timeout, open_new_tabs=args.new_tabs_only, attach_to_existing_chrome=args.chrome_remote_debug)
# await veb.start_browser()
# id_ = await veb.new_page()
# if not args.keep_chrome:
#     await veb.close_browser()

#  if args.chrome_headless:
#         asyncio.get_event_loop().run_until_complete(main_headless(args))

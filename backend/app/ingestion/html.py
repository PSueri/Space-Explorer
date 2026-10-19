from bs4 import BeautifulSoup

TRAILING_NAVIGATION_HEADINGS = frozenset(
    {
        "discover more topics from nasa",
    }
)


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")

    if main is None:
        raise ValueError("The page does not contain a main content element.")

    for element in main.select(
        "script, style, noscript, nav, header, footer, aside, form, button, svg"
    ):
        element.decompose()

    blocks: list[str] = []

    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        if element.find_parent(["p", "li"]) is not None:
            continue

        text = " ".join(element.get_text(" ", strip=True).split())

        if (
            element.name in {"h1", "h2", "h3", "h4"}
            and text.casefold() in TRAILING_NAVIGATION_HEADINGS
        ):
            break

        if text:
            blocks.append(text)

    text = "\n\n".join(blocks)

    if len(text) < 100:
        raise ValueError("The extracted main content is unexpectedly short.")

    return text

import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag, unquote
from typing import List, Dict, Any, Optional

from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoundingBox":
        return cls(
            x=int(d.get("x", 0)),
            y=int(d.get("y", 0)),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
        )

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


@dataclass
class ElementMetadata:
    tag: str
    id: str
    classes: str
    selector: str
    text: str
    is_interactive: bool
    role: str
    bbox: BoundingBox

    # DOM identity / hierarchy
    dom_id: int = -1
    element_index: int = -1
    parent_index: int = -1
    ancestor_indices: List[int] = field(default_factory=list)
    dom_depth: int = 0
    dom_order: int = 0
    child_count: int = 0
    sibling_index: int = 0

    # Document-space geometry
    doc_bbox: Optional[BoundingBox] = None
    viewport_width: int = 0
    viewport_height: int = 0
    visible_fraction: float = 1.0
    is_clipped_left: bool = False
    is_clipped_right: bool = False
    is_clipped_top: bool = False
    is_clipped_bottom: bool = False

    # Content / semantics
    full_text: str = ""
    has_direct_text: bool = False
    accessible_name: str = ""
    aria_label: str = ""
    aria_labelledby: str = ""
    aria_live: str = ""
    aria_hidden: bool = False
    alt: str = ""
    title: str = ""
    placeholder: str = ""
    href: str = ""
    input_type: str = ""
    required: bool = False
    disabled: bool = False
    readonly: bool = False
    tab_index: int = 0
    heading_level: int = 0
    is_css_truncated: bool = False
    text_lines: List[Dict[str, Any]] = field(default_factory=list)

    # Visual style (computed)
    computed_style: Dict[str, str] = field(default_factory=dict)
    has_background_image: bool = False
    has_background_color: bool = False
    has_border: bool = False
    has_shadow: bool = False
    is_position_fixed: bool = False
    is_structural_box: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bbox"] = self.bbox.to_dict()
        if self.doc_bbox is not None:
            d["doc_bbox"] = self.doc_bbox.to_dict()
        return d


@dataclass
class ScreenCapture:
    screen_id: str
    url: str
    page_name: str
    view_mode: str  # "desktop" or "mobile"
    scroll_index: int
    scroll_y: int
    viewport_width: int
    viewport_height: int
    image_path: str
    elements: List[ElementMetadata] = field(default_factory=list)

    document_height: int = 0
    document_width: int = 0
    slice_top: int = 0
    slice_bottom: int = 0
    device_scale_factor: float = 1.0

    def to_dict(self, include_elements: bool = True) -> Dict[str, Any]:
        d = {
            "screen_id": self.screen_id,
            "url": self.url,
            "page_name": self.page_name,
            "view_mode": self.view_mode,
            "scroll_index": self.scroll_index,
            "scroll_y": self.scroll_y,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "document_height": self.document_height,
            "document_width": self.document_width,
            "slice_top": self.slice_top,
            "slice_bottom": self.slice_bottom,
            "device_scale_factor": self.device_scale_factor,
            "image_path": self.image_path,
        }
        if include_elements:
            d["elements"] = [el.to_dict() for el in self.elements]
        return d


class WebAppCrawler:
    """
    Crawls local web applications (URLs or local files), capturing
    viewport-by-viewport screenshots and DOM element geometry in both
    Desktop and Mobile views across subpages.

    The DOM extraction intentionally records document coordinates, DOM
    ancestry, computed styles, text and accessibility attributes so the
    downstream analyzers can reason about layout facts rather than only
    saliency.
    """

    DESKTOP_VIEWPORT = {"width": 1920, "height": 1080}
    MOBILE_VIEWPORT = {"width": 390, "height": 844}
    MOBILE_USER_AGENT = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    )

    MAX_ELEMENTS = 6000

    DOM_EXTRACTION_SCRIPT = """
    () => {
        const elements = [];
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const scrollX = window.scrollX || window.pageXOffset || 0;
        const scrollY = window.scrollY || window.pageYOffset || 0;
        const MAX_ELEMENTS = 6000;

        const allNodes = Array.from(document.querySelectorAll('*'));
        const idMap = new Map();
        allNodes.forEach((el, i) => idMap.set(el, i));

        function getSelector(el) {
            if (el.id) return '#' + CSS.escape(el.id);
            const testId = el.getAttribute && el.getAttribute('data-testid');
            if (testId) return '[data-testid="' + testId.replace(/"/g, '\\\\"') + '"]';
            let path = [];
            let curr = el;
            while (curr && curr.nodeType === Node.ELEMENT_NODE && curr !== document.body) {
                let sel = curr.nodeName.toLowerCase();
                if (curr.id) {
                    sel += '#' + CSS.escape(curr.id);
                    path.unshift(sel);
                    break;
                } else {
                    let sib = curr, nth = 1;
                    while ((sib = sib.previousElementSibling)) {
                        if (sib.nodeName.toLowerCase() === curr.nodeName.toLowerCase()) nth++;
                    }
                    if (nth !== 1) sel += ':nth-of-type(' + nth + ')';
                }
                path.unshift(sel);
                curr = curr.parentNode;
                if (path.length >= 6) break;
            }
            return path.join(' > ');
        }

        function isSelectorUnique(sel) {
            try { return document.querySelectorAll(sel).length === 1; }
            catch (e) { return false; }
        }

        function getAncestorPath(el) {
            const path = [];
            let curr = el;
            while (curr && curr !== document.documentElement) {
                const id = idMap.get(curr);
                if (id !== undefined) path.push(id);
                curr = curr.parentElement;
            }
            return path.reverse();
        }

        function cssEscapeId(id) {
            try { return CSS.escape(id); } catch (e) { return id; }
        }

        function lineBoxesFor(el) {
            // Word-level rects grouped into the rendered lines. This is what lets
            // the audit see "one word dangling on its own line", which an element
            // box alone cannot show. Coordinates are document space (like doc_bbox).
            // Only words the element itself renders (direct or inline descendants)
            // count, so a wrapper is never attributed to its block children's text.
            const words = [];
            try {
                const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
                    acceptNode: (n) => {
                        let p = n.parentElement;
                        while (p && p !== el) {
                            const tag = p.tagName.toLowerCase();
                            const disp = (window.getComputedStyle(p) || {}).display || '';
                            if (disp.startsWith('inline') || disp === 'contents' ||
                                ['a','b','i','em','strong','span','small','u','sup','sub','code',
                                 'mark','abbr','cite','q','time','kbd','s','del','ins','big','tt'].includes(tag)) {
                                p = p.parentElement;
                                continue;
                            }
                            return NodeFilter.FILTER_REJECT;
                        }
                        return (n.nodeValue || '').trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
                    },
                });
                let node;
                while ((node = walker.nextNode())) {
                    const value = node.nodeValue || '';
                    const re = /\\S+/g;
                    let m;
                    while ((m = re.exec(value)) !== null) {
                        const range = document.createRange();
                        range.setStart(node, m.index);
                        range.setEnd(node, m.index + m[0].length);
                        const rects = range.getClientRects();
                        if (!rects || !rects.length) continue;
                        const r = rects[0];
                        if (r.width <= 0 || r.height <= 0) continue;
                        words.push({ x: r.left, y: r.top, w: r.width, h: r.height, t: m[0].slice(0, 24) });
                        if (words.length >= 400) break;
                    }
                    if (words.length >= 400) break;
                }
            } catch (e) { return []; }
            if (words.length < 2) return [];
            words.sort((a, b) => (a.y - b.y) || (a.x - b.x));
            const lines = [];
            for (const wd of words) {
                const cy = wd.y + wd.h / 2;
                const line = lines[lines.length - 1];
                if (line && Math.abs(cy - line.cy) <= Math.max(3, wd.h * 0.5)) {
                    line.words.push(wd);
                    line.cy = (line.cy * (line.words.length - 1) + cy) / line.words.length;
                    line.y1 = Math.max(line.y1, wd.y + wd.h);
                } else {
                    lines.push({ cy: cy, y1: wd.y + wd.h, words: [wd] });
                }
            }
            return lines.map(l => ({
                x: Math.round(Math.min(...l.words.map(w => w.x)) + scrollX),
                y: Math.round(Math.min(...l.words.map(w => w.y)) + scrollY),
                width: Math.round(Math.max(...l.words.map(w => w.x + w.w)) - Math.min(...l.words.map(w => w.x))),
                height: Math.round(l.y1 - Math.min(...l.words.map(w => w.y))),
                words: l.words.length,
                text: l.words.map(w => w.t).join(' ').slice(0, 120),
            }));
        }

        const interactiveTags = new Set(['a', 'button', 'input', 'select', 'textarea', 'summary']);
        const skipTags = new Set(['script', 'style', 'noscript', 'meta', 'link', 'head', 'html', 'body', 'template']);

        for (const el of allNodes) {
            if (elements.length >= MAX_ELEMENTS) break;
            const tag = el.tagName.toLowerCase();
            if (skipTags.has(tag)) continue;

            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) < 0.05) continue;

            const rect = el.getBoundingClientRect();
            if (rect.width <= 2 || rect.height <= 2) continue;
            if (rect.right <= 0 || rect.bottom <= 0 || rect.left >= vw || rect.top >= vh) continue;

            const vx1 = Math.max(0, Math.round(rect.left));
            const vy1 = Math.max(0, Math.round(rect.top));
            const vx2 = Math.min(vw, Math.round(rect.right));
            const vy2 = Math.min(vh, Math.round(rect.bottom));
            const w = vx2 - vx1;
            const h = vy2 - vy1;
            if (w <= 2 || h <= 2) continue;

            const isInteractive = interactiveTags.has(tag) ||
                                  ['button', 'link', 'checkbox', 'radio', 'switch', 'tab', 'menuitem', 'option'].includes(el.getAttribute('role')) ||
                                  el.onclick !== null ||
                                  style.cursor === 'pointer';

            const directText = Array.from(el.childNodes)
                .filter(n => n.nodeType === Node.TEXT_NODE)
                .map(n => n.textContent.trim())
                .filter(t => t.length > 0)
                .join(' ');
            const fullText = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
            const textValue = (directText || fullText).replace(/\\s+/g, ' ').trim();
            const textLines = isInteractive || textValue.split(/\\s+/).length < 2 ? [] : lineBoxesFor(el);

            const ariaLabel = el.getAttribute('aria-label') || '';
            const labelledBy = el.getAttribute('aria-labelledby') || '';
            let labelledByText = '';
            if (labelledBy) {
                labelledByText = labelledBy.split(/\\s+/)
                    .map(id => { const n = document.getElementById(id); return n ? (n.innerText || n.textContent || '').trim() : ''; })
                    .filter(Boolean).join(' ');
            }
            const alt = el.getAttribute('alt') || '';
            const title = el.getAttribute('title') || '';
            const placeholder = el.getAttribute('placeholder') || '';
            const accessibleName = (ariaLabel || labelledByText || alt || title || placeholder ||
                                     (textValue ? textValue.slice(0, 200) : '')).replace(/\\s+/g, ' ').trim();

            const classAndId = (typeof el.className === 'string' ? el.className : '') + ' ' + (el.id || '');
            const boxKeywords = ['card', 'grid', 'box', 'panel', 'tile', 'widget', 'container'];
            const hasBoxIndicator = boxKeywords.some(kw => classAndId.toLowerCase().includes(kw));
            const borderWidth = parseFloat(style.borderTopWidth) + parseFloat(style.borderRightWidth) +
                                parseFloat(style.borderBottomWidth) + parseFloat(style.borderLeftWidth);
            const hasBorder = borderWidth > 0 && style.borderTopStyle !== 'none';
            const hasShadow = style.boxShadow && style.boxShadow !== 'none';
            const hasBgImage = style.backgroundImage && style.backgroundImage !== 'none';
            const bgColor = style.backgroundColor || '';
            const transparentBg = !bgColor || bgColor === 'rgba(0, 0, 0, 0)' || bgColor === 'transparent';
            const isStructuralBox = Boolean(
                hasBoxIndicator || hasBorder || hasShadow || hasBgImage || !transparentBg ||
                ['article', 'section', 'nav', 'header', 'footer', 'form', 'fieldset', 'aside', 'main'].includes(tag)
            );

            const docX = Math.round(rect.left + scrollX);
            const docY = Math.round(rect.top + scrollY);
            const fullW = Math.round(rect.width);
            const fullH = Math.round(rect.height);
            const fullArea = Math.max(1, fullW * fullH);
            const visibleFraction = Math.min(1, Math.max(0, (w * h) / fullArea));

            const headingMatch = /^h([1-6])$/.exec(tag);
            const headingLevel = headingMatch ? parseInt(headingMatch[1], 10) : 0;

            const computed = {
                fontSize: style.fontSize || '',
                fontWeight: style.fontWeight || '',
                fontFamily: style.fontFamily || '',
                color: style.color || '',
                backgroundColor: bgColor,
                lineHeight: style.lineHeight || '',
                textAlign: style.textAlign || '',
                textTransform: style.textTransform || '',
                letterSpacing: style.letterSpacing || '',
                display: style.display || '',
                position: style.position || '',
                overflow: style.overflow || '',
                padding: style.padding || '',
                margin: style.margin || '',
                gap: style.gap || '',
                borderRadius: style.borderRadius || '',
                borderTopWidth: style.borderTopWidth || '',
                boxShadow: style.boxShadow || '',
                backgroundImage: style.backgroundImage || '',
                opacity: style.opacity || '',
                zIndex: style.zIndex || '',
                cursor: style.cursor || '',
            };

            const parentEl = el.parentElement;
            const parentDomId = parentEl ? (idMap.get(parentEl) ?? -1) : -1;
            const ancestorPath = getAncestorPath(el);
            const selector = getSelector(el);
            const selectorUnique = isSelectorUnique(selector);

            const siblingIndex = Array.from(el.parentElement ? el.parentElement.children : []).indexOf(el);

            elements.push({
                tag: tag,
                id: el.id || '',
                classes: Array.from(el.classList).join(' '),
                selector: selector,
                selector_unique: Boolean(selectorUnique),
                text: textValue.slice(0, 200),
                full_text: fullText.slice(0, 1000),
                has_direct_text: Boolean(directText),
                accessible_name: accessibleName.slice(0, 300),
                aria_label: ariaLabel,
                aria_labelledby: labelledBy,
                aria_live: el.getAttribute('aria-live') || '',
                aria_hidden: el.getAttribute('aria-hidden') === 'true',
                alt: alt,
                title: title,
                placeholder: placeholder,
                href: el.getAttribute('href') || '',
                input_type: (el.getAttribute('type') || '').toLowerCase(),
                required: Boolean(el.required),
                disabled: Boolean(el.disabled),
                readonly: Boolean(el.readOnly),
                tab_index: typeof el.tabIndex === 'number' ? el.tabIndex : 0,
                heading_level: headingLevel,
                is_css_truncated: Boolean(el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1),
                text_lines: textLines,
                is_interactive: Boolean(isInteractive),
                is_structural_box: isStructuralBox,
                has_background_image: Boolean(hasBgImage),
                has_background_color: Boolean(!transparentBg),
                has_border: Boolean(hasBorder),
                has_shadow: Boolean(hasShadow),
                is_position_fixed: Boolean(style.position === 'fixed' || style.position === 'sticky'),
                role: el.getAttribute('role') || '',
                dom_id: idMap.get(el) ?? -1,
                parent_dom_id: parentDomId,
                ancestor_dom_ids: ancestorPath,
                dom_depth: ancestorPath.length,
                dom_order: idMap.get(el) ?? 0,
                child_count: el.children.length,
                sibling_index: siblingIndex,
                viewport_width: vw,
                viewport_height: vh,
                visible_fraction: visibleFraction,
                is_clipped_left: rect.left < 0,
                is_clipped_right: rect.right > vw,
                is_clipped_top: rect.top < 0,
                is_clipped_bottom: rect.bottom > vh,
                computed_style: computed,
                bbox: { x: vx1, y: vy1, width: w, height: h },
                doc_bbox: { x: docX, y: docY, width: fullW, height: fullH },
            });
        }

        return elements;
    }
    """

    def __init__(self, headless: bool = True):
        self.headless = headless

    # ------------------------------------------------------------------ #
    # Target handling
    # ------------------------------------------------------------------ #
    def _normalize_target(self, target: str) -> str:
        target_str = str(target).strip()
        if target_str.startswith(("http://", "https://", "file://")):
            return target_str

        abs_path = os.path.abspath(target_str)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Target local file does not exist: {abs_path}")
        return Path(abs_path).as_uri()

    def validate_target(self, target: str, allow_remote: bool = False, project_root: Optional[str] = None) -> str:
        """
        Validates and normalizes a target. By default only local files under
        project_root and localhost URLs are allowed (prevents SSRF / arbitrary
        local file reads when the tool is exposed to an agent).
        """
        start_url = self._normalize_target(target)
        parsed = urlparse(start_url)

        if parsed.scheme == "file":
            path = os.path.abspath(unquote(parsed.path))
            if project_root:
                root = os.path.abspath(project_root)
                if os.path.commonpath([path, root]) != root:
                    raise PermissionError(
                        f"Target file is outside the project root: {path} (root: {root}). "
                        f"Use --allow-remote to override."
                    )
            return start_url

        if allow_remote:
            return start_url

        host = (parsed.hostname or "").lower()
        if host not in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
            raise PermissionError(
                f"Remote target '{host}' is blocked by default. Use --allow-remote to allow it."
            )
        return start_url

    def _is_internal_url(self, base_url: str, candidate_url: str) -> bool:
        base_parsed = urlparse(base_url)
        cand_parsed = urlparse(candidate_url)

        if cand_parsed.scheme not in ("http", "https", "file"):
            return False

        if base_parsed.scheme == "file":
            if cand_parsed.scheme != "file":
                return False
            cand_path = unquote(cand_parsed.path)
            return os.path.exists(cand_path)

        if base_parsed.netloc != cand_parsed.netloc:
            return False

        return True

    def _slugify(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", text)
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug or "index"

    # ------------------------------------------------------------------ #
    # Crawl
    # ------------------------------------------------------------------ #
    def crawl(
        self,
        target: str,
        output_dir: str = "output_screens",
        max_pages: int = 10,
        scroll_step_ratio: float = 0.85,
        views: Optional[List[str]] = None,
        allow_remote: bool = False,
        project_root: Optional[str] = None,
        page_timeout_ms: int = 20000,
        settle_ms: int = 250,
    ) -> List[ScreenCapture]:
        start_url = self.validate_target(target, allow_remote=allow_remote, project_root=project_root)
        views = views or ["desktop", "mobile"]
        os.makedirs(output_dir, exist_ok=True)

        screens: List[ScreenCapture] = []

        print(f"[WebAppCrawler] Starting crawl for: {start_url}")
        print(f"[WebAppCrawler] Screenshots will be saved to: {output_dir}")

        with sync_playwright() as p:
            browser: Browser = p.chromium.launch(headless=self.headless)
            try:
                discovered_pages = self._discover_pages(browser, start_url, max_pages, page_timeout_ms)
                print(f"[WebAppCrawler] Discovered {len(discovered_pages)} page(s): {discovered_pages}")

                for view_mode in views:
                    print(f"\n[WebAppCrawler] === Capturing View: {view_mode.upper()} ===")
                    context = self._create_context(browser, view_mode)
                    try:
                        page = context.new_page()
                        page.set_default_timeout(page_timeout_ms)
                        for page_url in discovered_pages:
                            url_clean, _ = urldefrag(page_url)
                            page_name = self._get_page_name(url_clean, start_url)
                            print(f"[WebAppCrawler] Navigating to: {url_clean} ({view_mode})")
                            try:
                                self._load_page(page, url_clean, page_timeout_ms)
                            except Exception as e:
                                print(f"[WebAppCrawler] Warning: Failed to load {url_clean}: {e}")
                                continue

                            page_screens = self._capture_page_screens(
                                page=page,
                                page_url=url_clean,
                                page_name=page_name,
                                view_mode=view_mode,
                                output_dir=output_dir,
                                scroll_step_ratio=scroll_step_ratio,
                                settle_ms=settle_ms,
                            )
                            screens.extend(page_screens)
                    finally:
                        context.close()
            finally:
                browser.close()

        print(f"\n[WebAppCrawler] Crawl completed! Total screens captured: {len(screens)}")
        return screens

    def _load_page(self, page: Page, url: str, timeout_ms: int) -> None:
        page.goto(url, wait_until="load", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=min(5000, timeout_ms))
        except Exception:
            pass
        try:
            page.evaluate("() => (document.fonts && document.fonts.ready) ? document.fonts.ready.then(() => true) : true")
        except Exception:
            pass
        # Force animations/transitions off for deterministic screenshots.
        try:
            page.add_style_tag(content="*, *::before, *::after { animation-duration: 0s !important; animation-delay: 0s !important; transition-duration: 0s !important; transition-delay: 0s !important; scroll-behavior: auto !important; }")
        except Exception:
            pass
        try:
            page.emulate_media(reduced_motion="reduce")
        except Exception:
            pass
        try:
            page.evaluate("""() => Promise.all(Array.from(document.images).map(img => {
                if (img.complete) return true;
                return new Promise(resolve => { img.addEventListener('load', resolve, {once:true}); img.addEventListener('error', resolve, {once:true}); });
            }))""")
        except Exception:
            pass

    def _discover_pages(self, browser: Browser, start_url: str, max_pages: int, timeout_ms: int) -> List[str]:
        discovered: List[str] = []
        queue = [start_url]
        seen = set()

        context = browser.new_context(viewport=self.DESKTOP_VIEWPORT)
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        try:
            while queue and len(discovered) < max_pages:
                curr_url = queue.pop(0)
                clean_url, _ = urldefrag(curr_url)
                if clean_url in seen:
                    continue
                seen.add(clean_url)
                discovered.append(clean_url)

                try:
                    self._load_page(page, clean_url, timeout_ms)
                    links = page.evaluate("""
                        () => Array.from(document.querySelectorAll('a[href]'))
                                   .map(a => a.href)
                                   .filter(h => h && !h.startsWith('mailto:') && !h.startsWith('tel:') && !h.startsWith('javascript:'))
                    """)
                    for link in links:
                        clean_link, _ = urldefrag(link)
                        if clean_link not in seen and self._is_internal_url(start_url, clean_link):
                            queue.append(clean_link)
                except Exception as e:
                    print(f"[WebAppCrawler] Page discovery note on {curr_url}: {e}")
        finally:
            context.close()

        return discovered

    def _create_context(self, browser: Browser, view_mode: str) -> BrowserContext:
        if view_mode == "mobile":
            return browser.new_context(
                viewport=self.MOBILE_VIEWPORT,
                user_agent=self.MOBILE_USER_AGENT,
                is_mobile=True,
                has_touch=True,
                device_scale_factor=1.0,
                reduced_motion="reduce",
            )
        return browser.new_context(
            viewport=self.DESKTOP_VIEWPORT,
            is_mobile=False,
            has_touch=False,
            device_scale_factor=1.0,
            reduced_motion="reduce",
        )

    def _get_page_name(self, url: str, start_url: str) -> str:
        parsed = urlparse(url)
        path = unquote(parsed.path)
        if not path or path == "/":
            return "index"
        name = os.path.basename(path.rstrip("/"))
        name = os.path.splitext(name)[0]
        return self._slugify(name) or "page"

    def _capture_page_screens(
        self,
        page: Page,
        page_url: str,
        page_name: str,
        view_mode: str,
        output_dir: str,
        scroll_step_ratio: float,
        settle_ms: int = 250,
    ) -> List[ScreenCapture]:
        captured: List[ScreenCapture] = []

        page.evaluate("window.scrollTo(0, 0);")
        page.wait_for_timeout(100)

        dimensions = page.evaluate("""
            () => ({
                scrollHeight: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight),
                scrollWidth: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
                innerHeight: window.innerHeight,
                innerWidth: window.innerWidth,
                dpr: window.devicePixelRatio || 1
            })
        """)

        scroll_height = int(dimensions["scrollHeight"])
        inner_height = int(dimensions["innerHeight"])
        inner_width = int(dimensions["innerWidth"])
        doc_width = int(dimensions["scrollWidth"])
        dpr = float(dimensions.get("dpr", 1.0))

        scroll_step = max(100, int(inner_height * scroll_step_ratio))
        scroll_positions = [0]
        if scroll_height > inner_height:
            curr_y = scroll_step
            while curr_y < scroll_height - (inner_height * 0.2):
                scroll_positions.append(curr_y)
                curr_y += scroll_step
            max_scroll = scroll_height - inner_height
            if max_scroll > scroll_positions[-1] + 50:
                scroll_positions.append(max_scroll)

        print(
            f"[WebAppCrawler] Page '{page_name}' ({view_mode}): "
            f"scrollHeight={scroll_height}px, capturing {len(scroll_positions)} slice(s)"
        )

        for idx, scroll_y in enumerate(scroll_positions):
            page.evaluate(f"window.scrollTo(0, {int(scroll_y)});")
            page.wait_for_timeout(settle_ms)

            screen_id = f"{view_mode}_{page_name}_s{idx}"
            image_path = os.path.join(output_dir, f"{screen_id}.png")
            page.screenshot(path=image_path, full_page=False)

            raw_elements = page.evaluate(self.DOM_EXTRACTION_SCRIPT)
            element_objects: List[ElementMetadata] = []
            dom_id_to_index: Dict[int, int] = {}

            for raw in raw_elements:
                bbox_data = raw.get("bbox", {})
                doc_bbox_data = raw.get("doc_bbox", {})
                bbox = BoundingBox.from_dict(bbox_data)
                doc_bbox = BoundingBox.from_dict(doc_bbox_data) if doc_bbox_data else BoundingBox(
                    x=bbox.x + int(scroll_y),
                    y=bbox.y + int(scroll_y),
                    width=bbox.width,
                    height=bbox.height,
                )
                el_meta = ElementMetadata(
                    tag=raw.get("tag", ""),
                    id=raw.get("id", ""),
                    classes=raw.get("classes", ""),
                    selector=raw.get("selector", ""),
                    text=raw.get("text", ""),
                    full_text=raw.get("full_text", ""),
                    has_direct_text=bool(raw.get("has_direct_text", False)),
                    accessible_name=raw.get("accessible_name", ""),
                    aria_label=raw.get("aria_label", ""),
                    aria_labelledby=raw.get("aria_labelledby", ""),
                    aria_live=raw.get("aria_live", ""),
                    aria_hidden=bool(raw.get("aria_hidden", False)),
                    alt=raw.get("alt", ""),
                    title=raw.get("title", ""),
                    placeholder=raw.get("placeholder", ""),
                    href=raw.get("href", ""),
                    input_type=raw.get("input_type", ""),
                    required=bool(raw.get("required", False)),
                    disabled=bool(raw.get("disabled", False)),
                    readonly=bool(raw.get("readonly", False)),
                    tab_index=int(raw.get("tab_index", 0) or 0),
                    heading_level=int(raw.get("heading_level", 0) or 0),
                    is_css_truncated=bool(raw.get("is_css_truncated", False)),
                    text_lines=list(raw.get("text_lines", []) or []),
                    is_interactive=bool(raw.get("is_interactive", False)),
                    is_structural_box=bool(raw.get("is_structural_box", False)),
                    has_background_image=bool(raw.get("has_background_image", False)),
                    has_background_color=bool(raw.get("has_background_color", False)),
                    has_border=bool(raw.get("has_border", False)),
                    has_shadow=bool(raw.get("has_shadow", False)),
                    is_position_fixed=bool(raw.get("is_position_fixed", False)),
                    role=raw.get("role", ""),
                    bbox=bbox,
                    doc_bbox=doc_bbox,
                    dom_id=int(raw.get("dom_id", -1)),
                    dom_depth=int(raw.get("dom_depth", 0) or 0),
                    dom_order=int(raw.get("dom_order", 0) or 0),
                    child_count=int(raw.get("child_count", 0) or 0),
                    sibling_index=int(raw.get("sibling_index", 0) or 0),
                    viewport_width=int(raw.get("viewport_width", inner_width) or inner_width),
                    viewport_height=int(raw.get("viewport_height", inner_height) or inner_height),
                    visible_fraction=float(raw.get("visible_fraction", 1.0) or 1.0),
                    is_clipped_left=bool(raw.get("is_clipped_left", False)),
                    is_clipped_right=bool(raw.get("is_clipped_right", False)),
                    is_clipped_top=bool(raw.get("is_clipped_top", False)),
                    is_clipped_bottom=bool(raw.get("is_clipped_bottom", False)),
                    computed_style=raw.get("computed_style", {}) or {},
                )
                dom_id_to_index[el_meta.dom_id] = len(element_objects)
                element_objects.append(el_meta)

            # Resolve DOM ancestry to indices of extracted elements.
            raw_by_dom_id = {int(r.get("dom_id", -1)): r for r in raw_elements}
            for idx_el, el in enumerate(element_objects):
                el.element_index = idx_el
                raw = raw_by_dom_id.get(el.dom_id, {})
                parent_dom_id = int(raw.get("parent_dom_id", -1) or -1)
                ancestor_dom_ids = raw.get("ancestor_dom_ids", []) or []

                ancestors: List[int] = []
                for dom_id in ancestor_dom_ids:
                    if dom_id in dom_id_to_index and dom_id_to_index[dom_id] != idx_el:
                        ancestors.append(dom_id_to_index[dom_id])
                el.ancestor_indices = ancestors
                el.parent_index = ancestors[-1] if ancestors else -1

            screen_capture = ScreenCapture(
                screen_id=screen_id,
                url=page_url,
                page_name=page_name,
                view_mode=view_mode,
                scroll_index=idx,
                scroll_y=int(scroll_y),
                viewport_width=inner_width,
                viewport_height=inner_height,
                image_path=image_path,
                elements=element_objects,
                document_height=scroll_height,
                document_width=doc_width,
                slice_top=int(scroll_y),
                slice_bottom=int(scroll_y) + inner_height,
                device_scale_factor=dpr,
            )
            captured.append(screen_capture)

        return captured


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Local Web App Crawler & Screenshot Taker")
    parser.add_argument("target", help="Target URL or local HTML file path")
    parser.add_argument("--output", default="output_screens", help="Output directory for screenshots")
    parser.add_argument("--max-pages", type=int, default=5, help="Maximum number of subpages to crawl")
    parser.add_argument("--allow-remote", action="store_true", help="Allow non-localhost URLs")
    args = parser.parse_args()

    crawler = WebAppCrawler(headless=True)
    results = crawler.crawl(
        args.target,
        output_dir=args.output,
        max_pages=args.max_pages,
        allow_remote=args.allow_remote,
        project_root=os.getcwd(),
    )
    print(f"Captured {len(results)} screens in total.")

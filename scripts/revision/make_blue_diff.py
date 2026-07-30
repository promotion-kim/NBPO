"""Rebuild main_v3.tex as a professor-review copy: paragraph blocks that differ
from main.tex are wrapped in {\revblue ...} (blue), unchanged blocks stay
black, and every \kekim{...} comment from main.tex is re-inserted at the
position corresponding to its original location. Deleted-only text is dropped
(its kekim comment still anchors at the deletion site). Paragraph-level
granularity keeps the marked file compilable.
"""
import difflib, re, sys

ORIG = "main.tex"
NEW = sys.argv[1] if len(sys.argv) > 1 else "main_v3.tex"


def split_preamble(text):
    i = text.index("\\begin{document}")
    return text[:i], text[i:]


def blocks(body):
    return [b for b in re.split(r"\n\s*\n", body) if b.strip()]


def extract_kekim(block):
    """Remove \\kekim{...} (balanced braces) from a block; return (clean, [comments])."""
    out, comments, i = [], [], 0
    while True:
        j = block.find("\\kekim{", i)
        if j < 0:
            out.append(block[i:])
            break
        out.append(block[i:j])
        depth, k = 1, j + len("\\kekim{")
        while depth and k < len(block):
            if block[k] == "{": depth += 1
            elif block[k] == "}": depth -= 1
            k += 1
        comments.append(block[j:k])
        i = k
    return "".join(out).strip(), comments


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    orig_pre, orig_body = split_preamble(open(ORIG, encoding="utf-8").read())
    new_pre, new_body = split_preamble(open(NEW, encoding="utf-8").read())

    orig_blocks_raw = blocks(orig_body)
    orig_blocks, kekim_at = [], {}          # kekim_at[idx of clean orig block] = [comments]
    for b in orig_blocks_raw:
        clean, comments = extract_kekim(b)
        if clean:
            orig_blocks.append(clean)
            if comments:
                kekim_at.setdefault(len(orig_blocks) - 1, []).extend(comments)
        elif comments:                       # comment-only block: anchor to previous block
            kekim_at.setdefault(max(len(orig_blocks) - 1, 0), []).extend(comments)

    new_blocks = blocks(new_body)
    sm = difflib.SequenceMatcher(a=[norm(b) for b in orig_blocks],
                                 b=[norm(b) for b in new_blocks], autojunk=False)

    out, n_blue, n_kek = [], 0, 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        pend = []                            # comments anchored in this orig range
        for oi in range(i1, i2):
            pend.extend(kekim_at.pop(oi, []))
        if tag == "equal":
            for oi, nj in zip(range(i1, i2), range(j1, j2)):
                out.append(new_blocks[nj])
                for c in kekim_at.pop(oi, []):
                    out.append(c); n_kek += 1
            pend = []
        else:                                # replace / insert / delete
            for nj in range(j1, j2):
                b = new_blocks[nj]
                # never wrap blocks whose effect must escape the group
                # (definitions/counters produce no visible output anyway)
                if (b.lstrip().startswith("\\end{document}")
                        or re.search(r"^\s*\\(newcommand|renewcommand|providecommand|setcounter|input)\b", b, re.M)):
                    out.append(b)
                else:
                    out.append("{\\revblue " + b + "\\par}")
                    n_blue += 1
            for c in pend:                   # anchor orig comments at the change site
                out.append(c); n_kek += 1
    for oi in sorted(kekim_at):              # any stragglers
        for c in kekim_at[oi]:
            out.append(c); n_kek += 1

    if "\\revblue" not in new_pre:
        new_pre = new_pre.replace(
            "\\newcommand{\\kekim}",
            "\\newcommand{\\revblue}{\\color{blue}}\n\\newcommand{\\kekim}", 1)
    open(NEW, "w", encoding="utf-8").write(new_pre + "\n\n".join(out) + "\n")
    print(f"[blue-diff] {NEW}: {n_blue} blue blocks, {n_kek} kekim comments re-inserted, "
          f"{len(new_blocks)} total blocks")


if __name__ == "__main__":
    main()

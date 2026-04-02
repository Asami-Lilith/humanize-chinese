#!/usr/bin/env python3
"""
verify_cn.py — 使用 Sapling AI API 交叉验证 AI 检测结果
Cross-verify AI detection results using Sapling AI API.

Usage:
    python verify_cn.py <file>                    # 验证单个文件
    python verify_cn.py <file> -o <outfile>       # 改写 + 验证
    python verify_cn.py <before> <after>          # 对比验证
    python verify_cn.py <file> --api-key KEY      # 指定 API key
"""

import sys
import os
import json
import argparse

# Try to import urllib (stdlib, no deps)
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

SAPLING_API_URL = "https://api.sapling.ai/api/v1/aidetect"

# Default key locations
KEY_LOCATIONS = [
    os.path.expanduser("~/.openclaw/.credentials/sapling"),
    os.path.expanduser("~/.config/sapling/key"),
]


def get_api_key(cli_key=None):
    """Get Sapling API key from CLI arg, env, or credential files."""
    if cli_key:
        return cli_key

    env_key = os.environ.get("SAPLING_API_KEY")
    if env_key:
        return env_key

    for path in KEY_LOCATIONS:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("SAPLING_API_KEY="):
                        return line.split("=", 1)[1].strip()
    return None


def sapling_detect(text, api_key, sent_scores=True):
    """Call Sapling AI detection API. Returns score 0-1."""
    payload = json.dumps({
        "key": api_key,
        "text": text,
        "sent_scores": sent_scores,
    }).encode("utf-8")

    req = Request(
        SAPLING_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"❌ Sapling API error ({e.code}): {body}", file=sys.stderr)
        return None
    except URLError as e:
        print(f"❌ Network error: {e.reason}", file=sys.stderr)
        return None


def score_to_level(score):
    """Convert 0-1 score to human-readable level."""
    pct = int(score * 100)
    if pct >= 75:
        return f"{pct}% 🔴 VERY HIGH"
    elif pct >= 50:
        return f"{pct}% 🟠 HIGH"
    elif pct >= 25:
        return f"{pct}% 🟡 MEDIUM"
    else:
        return f"{pct}% 🟢 LOW"


def print_result(label, result):
    """Pretty-print detection result."""
    if result is None:
        print(f"  {label}: ❌ API 调用失败")
        return

    score = result.get("score", 0)
    print(f"  {label}: {score_to_level(score)}")

    # Show top suspicious sentences
    sentences = result.get("sentence_scores", [])
    if sentences:
        # Sort by score descending
        top = sorted(sentences, key=lambda s: s.get("score", 0), reverse=True)[:3]
        suspicious = [s for s in top if s.get("score", 0) > 0.5]
        if suspicious:
            print(f"  最可疑的句子:")
            for s in suspicious:
                sent = s["sentence"][:60] + ("..." if len(s["sentence"]) > 60 else "")
                print(f"    [{int(s['score']*100)}%] {sent}")


def read_file(path):
    """Read text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser(
        description="使用 Sapling AI API 交叉验证 AI 检测结果"
    )
    parser.add_argument("files", nargs="+", help="要验证的文件 (1个=单独验证, 2个=对比验证)")
    parser.add_argument("--api-key", help="Sapling API key")
    parser.add_argument("-o", "--output", help="改写后的输出文件 (与 --rewrite 配合)")
    parser.add_argument("--rewrite", "-r", action="store_true",
                        help="先用 humanize_cn.py 改写，再验证")
    parser.add_argument("-a", "--aggressive", action="store_true",
                        help="激进改写模式")
    parser.add_argument("-j", "--json", action="store_true",
                        help="JSON 输出")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="只输出分数")

    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    if not api_key:
        print("❌ 找不到 Sapling API key", file=sys.stderr)
        print("  设置方法:", file=sys.stderr)
        print("  1. export SAPLING_API_KEY=your_key", file=sys.stderr)
        print("  2. 写入 ~/.openclaw/.credentials/sapling", file=sys.stderr)
        print("  3. 用 --api-key 参数", file=sys.stderr)
        sys.exit(1)

    if args.rewrite and len(args.files) == 1:
        # Rewrite + verify mode
        original_text = read_file(args.files[0])

        # Run local detection + rewrite
        script_dir = os.path.dirname(os.path.abspath(__file__))
        import subprocess

        out_path = args.output or args.files[0].rsplit(".", 1)[0] + "_clean.txt"
        cmd = [sys.executable, os.path.join(script_dir, "compare_cn.py"),
               args.files[0], "-o", out_path]
        if args.aggressive:
            cmd.append("-a")

        print("📝 本地改写中...")
        subprocess.run(cmd, check=True)

        rewritten_text = read_file(out_path)

        print("\n🔍 Sapling AI 交叉验证:")
        result_before = sapling_detect(original_text, api_key)
        result_after = sapling_detect(rewritten_text, api_key)

        if args.json:
            print(json.dumps({
                "original": {"sapling_score": result_before.get("score") if result_before else None},
                "rewritten": {"sapling_score": result_after.get("score") if result_after else None},
            }, ensure_ascii=False, indent=2))
        else:
            print_result("原文 (Sapling)", result_before)
            print_result("改写后 (Sapling)", result_after)

            if result_before and result_after:
                before_pct = int(result_before["score"] * 100)
                after_pct = int(result_after["score"] * 100)
                diff = before_pct - after_pct
                if diff > 0:
                    print(f"\n  ✅ Sapling 验证: 降低了 {diff} 个百分点")
                else:
                    print(f"\n  ⚠️ Sapling 检测分数未降低")

    elif len(args.files) == 2:
        # Compare mode: before and after
        text_before = read_file(args.files[0])
        text_after = read_file(args.files[1])

        print("🔍 Sapling AI 交叉验证:")
        result_before = sapling_detect(text_before, api_key)
        result_after = sapling_detect(text_after, api_key)

        if args.json:
            print(json.dumps({
                "before": {"file": args.files[0], "sapling_score": result_before.get("score") if result_before else None},
                "after": {"file": args.files[1], "sapling_score": result_after.get("score") if result_after else None},
            }, ensure_ascii=False, indent=2))
        else:
            print_result(f"文件1 ({args.files[0]})", result_before)
            print_result(f"文件2 ({args.files[1]})", result_after)

            if result_before and result_after:
                before_pct = int(result_before["score"] * 100)
                after_pct = int(result_after["score"] * 100)
                diff = before_pct - after_pct
                if diff > 0:
                    print(f"\n  ✅ 降低了 {diff} 个百分点")

    else:
        # Single file verify
        text = read_file(args.files[0])

        if args.quiet:
            result = sapling_detect(text, api_key, sent_scores=False)
            if result:
                print(int(result["score"] * 100))
            sys.exit(0)

        print(f"🔍 Sapling AI 检测: {args.files[0]}")
        result = sapling_detect(text, api_key)

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result("AI 概率", result)


if __name__ == "__main__":
    main()

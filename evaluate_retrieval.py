import argparse
import json
from pathlib import Path

from rag_pipeline import build_vector_database, retrieve_relevant_chunks


def normalize(text):
    return text.lower().strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval with keyword checks.")
    parser.add_argument("--pdf", required=True, help="PDF used to build the vector database")
    parser.add_argument(
        "--questions",
        default="evaluation/questions.json",
        help="Evaluation dataset in JSON format",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    build_vector_database(args.pdf)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    passed = 0
    evaluated = 0
    for item in questions:
        chunks = retrieve_relevant_chunks(item["question"], top_k=args.top_k)
        retrieved_text = normalize(" ".join(chunk["text"] for chunk in chunks))
        keywords = [normalize(keyword) for keyword in item["expected_keywords"]]

        if not keywords:
            print(f"[REVIEW] {item['id']}: out-of-scope case requires manual answer review")
            continue

        evaluated += 1
        matched = [keyword for keyword in keywords if keyword in retrieved_text]
        success = len(matched) == len(keywords)
        passed += int(success)
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {item['id']}: matched {len(matched)}/{len(keywords)} keywords")

    score = (passed / evaluated * 100) if evaluated else 0
    print(f"\nRetrieval keyword score: {passed}/{evaluated} ({score:.1f}%)")


if __name__ == "__main__":
    main()

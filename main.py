import argparse
import sys
from src.pipeline import AnonymizationPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Анонімізація транспортних даних через k-ан. + l-різн. + OWL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True,
                   help="Шлях до директорії датасету")
    p.add_argument("--output", default="./output",
                   help="Директорія для результатів")
    p.add_argument("--k", type=int, default=3,
                   help="Параметр k-анонімності (≥2)")
    p.add_argument("--l", type=int, default=2,
                   help="Параметр l-різноманітності (≥2)")
    p.add_argument("--geohash-precision", type=int, default=5,
                   dest="geohash_precision",
                   help="Точність геохешу 1–8 (5 → ~2.4km, 4 → ~20km)")
    p.add_argument("--time-bucket", type=int, default=300,
                   dest="time_bucket",
                   help="Розмір часового bucket у секундах")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.k < 2:
        print("Помилка: k має бути ≥ 2", file=sys.stderr)
        sys.exit(1)
    if args.l < 2:
        print("Помилка: l має бути ≥ 2", file=sys.stderr)
        sys.exit(1)
    if not (1 <= args.geohash_precision <= 8):
        print("Помилка: --geohash-precision має бути від 1 до 8", file=sys.stderr)
        sys.exit(1)

    pipeline = AnonymizationPipeline(
        dataset_path=args.dataset,
        output_path=args.output,
        k=args.k,
        l=args.l,
        geohash_precision=args.geohash_precision,
        time_bucket_sec=args.time_bucket,
    )

    try:
        report = pipeline.run()
        print(f"\n✓ Анонімізація завершена!")
        print(f"  Записів: {report.anonymized_records}/{report.original_records} "
              f"(пригнічено {report.suppression_rate:.1%})")
        print(f"  Результати: {args.output}/")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\n✗ Не знайдено датасет: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n✗ Помилка даних: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Неочікувана помилка: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\user\PycharmProjects\mastersThesis\UAH-DriveSet"
    OUTPUT_PATH  = r"C:\Users\user\PycharmProjects\mastersThesis\output"

    pipeline = AnonymizationPipeline(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        k=3,
        l=2,
        geohash_precision=5,
        time_bucket_sec=300,
    )

    try:
        report = pipeline.run()
        print(f"\n✓ Анонімізація завершена!")
        print(f"  Записів: {report.anonymized_records}/{report.original_records} "
              f"(пригнічено {report.suppression_rate:.1%})")
        print(f"  Результати: {OUTPUT_PATH}")
    except FileNotFoundError as e:
        print(f"\n✗ Не знайдено датасет: {e}")
    except ValueError as e:
        print(f"\n✗ Помилка даних: {e}")
        raise
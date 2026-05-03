import argparse
from pathlib import Path
import SimpleITK as sitk


def resample_to_reference(moving_path: Path, reference_path: Path, output_path: Path, is_label: bool) -> None:
    moving = sitk.ReadImage(str(moving_path))
    reference = sitk.ReadImage(str(reference_path))
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    aligned = sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        interpolator,
        0,
        moving.GetPixelID(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(aligned, str(output_path), useCompression=True)


def align_case(t2: Path, adc: Path, dwi: Path, gland: Path, cancer: Path, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    out_t2 = output_dir / f"{prefix}_t2.nii.gz"
    out_adc = output_dir / f"{prefix}_adc_to_t2.nii.gz"
    out_dwi = output_dir / f"{prefix}_dwi_to_t2.nii.gz"
    out_gland = output_dir / f"{prefix}_gland_to_t2.nii.gz"
    out_cancer = output_dir / f"{prefix}_cancer_to_t2.nii.gz"

    # Keep T2 as reference (copy as-is)
    sitk.WriteImage(sitk.ReadImage(str(t2)), str(out_t2), useCompression=True)
    resample_to_reference(adc, t2, out_adc, is_label=False)
    resample_to_reference(dwi, t2, out_dwi, is_label=False)
    resample_to_reference(gland, t2, out_gland, is_label=True)
    resample_to_reference(cancer, t2, out_cancer, is_label=True)

    print("[OK] Aligned outputs:")
    print(f"  T2     : {out_t2}")
    print(f"  ADC    : {out_adc}")
    print(f"  DWI    : {out_dwi}")
    print(f"  Gland  : {out_gland}")
    print(f"  Cancer : {out_cancer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align mpMRI modalities and labels to T2 reference grid.")
    parser.add_argument("--t2", required=True, help="Path to T2 NIfTI (.nii.gz)")
    parser.add_argument("--adc", required=True, help="Path to ADC NIfTI (.nii.gz)")
    parser.add_argument("--dwi", required=True, help="Path to DWI NIfTI (.nii.gz)")
    parser.add_argument("--gland", required=True, help="Path to gland mask NIfTI (.nii.gz)")
    parser.add_argument("--cancer", required=True, help="Path to cancer mask NIfTI (.nii.gz)")
    parser.add_argument("--output_dir", required=True, help="Output folder for aligned files")
    parser.add_argument("--prefix", default="case", help="Filename prefix for aligned outputs")
    args = parser.parse_args()

    align_case(
        t2=Path(args.t2),
        adc=Path(args.adc),
        dwi=Path(args.dwi),
        gland=Path(args.gland),
        cancer=Path(args.cancer),
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()

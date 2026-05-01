import argparse
from pathlib import Path
import SimpleITK as sitk


def convert_series(dicom_dir: Path, output_path: Path) -> None:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {dicom_dir}")
    if len(series_ids) > 1:
        print(f"[INFO] Found {len(series_ids)} series in {dicom_dir}; using first series ID: {series_ids[0]}")

    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    if not file_names:
        raise ValueError(f"No DICOM files found for series {series_ids[0]} in: {dicom_dir}")

    reader.SetFileNames(file_names)
    image = reader.Execute()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), useCompression=True)
    print(f"[OK] Wrote NIfTI: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a DICOM series folder to .nii.gz")
    parser.add_argument("--dicom_dir", required=True, help="Folder containing one DICOM series")
    parser.add_argument("--output", required=True, help="Output .nii.gz path")
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    output_path = Path(args.output)

    if not dicom_dir.exists() or not dicom_dir.is_dir():
        raise ValueError(f"dicom_dir does not exist or is not a directory: {dicom_dir}")
    if output_path.suffixes[-2:] != ['.nii', '.gz']:
        raise ValueError("output must end with .nii.gz")

    convert_series(dicom_dir, output_path)


if __name__ == "__main__":
    main()

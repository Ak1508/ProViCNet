import argparse
from pathlib import Path
from typing import Optional

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


def convert_rtstruct_to_nifti_mask(
    dicom_dir: Path,
    rtstruct_path: Path,
    output_path: Path,
    roi_name: Optional[str] = None,
) -> None:
    """
    Convert an RTSTRUCT (RS*.dcm) to a binary NIfTI mask aligned to the referenced DICOM series.

    Requires `rt-utils` package:
      pip install rt-utils
    """
    try:
        from rt_utils import RTStructBuilder
    except ImportError as exc:
        raise ImportError(
            "RTSTRUCT conversion requires `rt-utils`. Install it with: pip install rt-utils"
        ) from exc

    if not rtstruct_path.exists() or not rtstruct_path.is_file():
        raise ValueError(f"rtstruct file does not exist: {rtstruct_path}")

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=str(dicom_dir),
        rt_struct_path=str(rtstruct_path),
    )

    roi_names = rtstruct.get_roi_names()
    if not roi_names:
        raise ValueError("No ROI found in RTSTRUCT")

    selected_roi = roi_name if roi_name else roi_names[0]
    if selected_roi not in roi_names:
        raise ValueError(f"ROI '{selected_roi}' not found. Available ROIs: {roi_names}")

    print(f"[INFO] Using ROI: {selected_roi}")
    mask = rtstruct.get_roi_mask_by_name(selected_roi).astype('uint8')

    # rt-utils returns mask as (x, y, z); SimpleITK expects (z, y, x)
    mask_zyx = mask.transpose(2, 1, 0)

    ref_reader = sitk.ImageSeriesReader()
    series_ids = ref_reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {dicom_dir}")
    ref_reader.SetFileNames(ref_reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0]))
    ref_image = ref_reader.Execute()

    mask_image = sitk.GetImageFromArray(mask_zyx)
    mask_image.CopyInformation(ref_image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(mask_image, str(output_path), useCompression=True)
    print(f"[OK] Wrote RTSTRUCT mask NIfTI: {output_path}")


def _validate_output_path(output_path: Path) -> None:
    if output_path.suffixes[-2:] != [".nii", ".gz"]:
        raise ValueError("output must end with .nii.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DICOM series / RTSTRUCT to .nii.gz")
    parser.add_argument("--dicom_dir", required=True, help="Folder containing the referenced DICOM image series")
    parser.add_argument("--output", required=True, help="Output .nii.gz path")
    parser.add_argument("--rtstruct", default=None, help="Optional RTSTRUCT RS.dcm path; when set, outputs ROI mask NIfTI")
    parser.add_argument("--roi_name", default=None, help="Optional ROI name from RTSTRUCT; default uses the first ROI")
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    output_path = Path(args.output)

    if not dicom_dir.exists() or not dicom_dir.is_dir():
        raise ValueError(f"dicom_dir does not exist or is not a directory: {dicom_dir}")
    _validate_output_path(output_path)

    if args.rtstruct:
        convert_rtstruct_to_nifti_mask(
            dicom_dir=dicom_dir,
            rtstruct_path=Path(args.rtstruct),
            output_path=output_path,
            roi_name=args.roi_name,
        )
    else:
        convert_series(dicom_dir, output_path)


if __name__ == "__main__":
    main()

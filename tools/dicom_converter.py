import os
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage import draw
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# RT STRUCT TO MASK CONVERSION
# ============================================================================

def extract_rtstruct_contours(rtstruct_path):
    """
    Extract contours from RT Struct DICOM file with support for multiple formats.
    Handles different RT Struct DICOM structures robustly.
    
    Returns:
        dict: Dictionary with structure names as keys and list of contours as values
    """
    rtstruct = pydicom.dcmread(rtstruct_path)
    contours_dict = {}
    
    if not hasattr(rtstruct, 'ROIContourSequence'):
        logger.warning("No ROI Contour Sequence found in RT Struct")
        return contours_dict
    
    # Method 1: Try standard mapping using StructureSetROISequence
    roi_to_name = {}
    if hasattr(rtstruct, 'StructureSetROISequence'):
        try:
            for roi_seq in rtstruct.StructureSetROISequence:
                roi_number = roi_seq.ReferencedROINumber
                roi_name = roi_seq.ROIName
                roi_to_name[roi_number] = roi_name
                logger.debug(f"ROI {roi_number}: {roi_name}")
        except Exception as e:
            logger.warning(f"Error building ROI mapping: {e}")
    
    logger.info(f"Found {len(roi_to_name)} ROI mappings")
    
    # Process contours
    for idx, roi_contour in enumerate(rtstruct.ROIContourSequence):
        try:
            structure_name = None
            
            # Try Method 1: Use ReferencedROINumber to look up name
            if hasattr(roi_contour, 'ReferencedROINumber'):
                roi_number = roi_contour.ReferencedROINumber
                if roi_number in roi_to_name:
                    structure_name = roi_to_name[roi_number]
            
            # Try Method 2: Some RT Structs have ReferencedROIName directly
            if structure_name is None and hasattr(roi_contour, 'ReferencedROIName'):
                structure_name = roi_contour.ReferencedROIName
            
            # Try Method 3: Try to find name in StructureSetROISequence by sequence index
            if structure_name is None and hasattr(rtstruct, 'StructureSetROISequence'):
                if idx < len(rtstruct.StructureSetROISequence):
                    try:
                        structure_name = rtstruct.StructureSetROISequence[idx].ROIName
                    except:
                        pass
            
            # If still no name, skip
            if structure_name is None:
                logger.warning(f"Could not determine structure name for contour {idx}")
                continue
            
            # Extract contours
            contours_list = []
            if hasattr(roi_contour, 'ContourSequence'):
                for contour_idx, contour in enumerate(roi_contour.ContourSequence):
                    try:
                        contour_data = contour.ContourData
                        # Convert flat list to (x, y, z) tuples
                        coords = [(float(contour_data[i]), float(contour_data[i+1]), float(contour_data[i+2])) 
                                 for i in range(0, len(contour_data), 3)]
                        
                        if coords:  # Only add non-empty contours
                            contours_list.append(coords)
                    except Exception as e:
                        logger.debug(f"Error reading contour {contour_idx} for {structure_name}: {e}")
                        continue
            
            if contours_list:
                contours_dict[structure_name] = contours_list
                logger.info(f"✓ Loaded '{structure_name}' with {len(contours_list)} contours")
            else:
                logger.warning(f"✗ No valid contours found for '{structure_name}'")
        
        except Exception as e:
            logger.warning(f"Error processing ROI contour {idx}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue
    
    return contours_dict


def print_rtstruct_structure(rtstruct_path):
    """
    Print the structure of RT Struct file for debugging.
    """
    rtstruct = pydicom.dcmread(rtstruct_path)
    
    print("\n" + "="*60)
    print("RT STRUCT FILE STRUCTURE")
    print("="*60)
    
    if hasattr(rtstruct, 'StructureSetROISequence'):
        print(f"\nStructureSetROISequence ({len(rtstruct.StructureSetROISequence)} items):")
        for i, roi in enumerate(rtstruct.StructureSetROISequence):
            print(f"  [{i}] ReferencedROINumber: {roi.ReferencedROINumber}, ROIName: {roi.ROIName}")
    
    if hasattr(rtstruct, 'ROIContourSequence'):
        print(f"\nROIContourSequence ({len(rtstruct.ROIContourSequence)} items):")
        for i, roi_contour in enumerate(rtstruct.ROIContourSequence):
            attrs = []
            if hasattr(roi_contour, 'ReferencedROINumber'):
                attrs.append(f"ReferencedROINumber: {roi_contour.ReferencedROINumber}")
            if hasattr(roi_contour, 'ReferencedROIName'):
                attrs.append(f"ReferencedROIName: {roi_contour.ReferencedROIName}")
            if hasattr(roi_contour, 'ContourSequence'):
                attrs.append(f"ContourSequence: {len(roi_contour.ContourSequence)} contours")
            print(f"  [{i}] {', '.join(attrs)}")
    
    print("="*60 + "\n")


def poly2mask(coords_x, coords_y, shape):
    """
    Convert polygon coordinates to binary mask.
    """
    try:
        mask = draw.polygon2mask(tuple(reversed(shape)), 
                               np.column_stack((coords_y, coords_x)))
        return mask
    except Exception as e:
        logger.error(f"Error creating polygon mask: {e}")
        return np.zeros(tuple(reversed(shape)), dtype=bool)


def contours_to_mask(contours_dict, dicom_image, structure_names=None):
    """
    Convert RT Struct contours to binary mask array.
    """
    shape = dicom_image.GetSize()  # (width, height, depth)
    
    # Initialize mask array (z, y, x) - SimpleITK convention
    mask = np.zeros((shape[2], shape[1], shape[0]), dtype=np.uint8)
    
    # Filter structures if specified
    structures_to_process = (structure_names 
                            if structure_names 
                            else list(contours_dict.keys()))
    
    logger.info(f"Processing structures: {structures_to_process}")
    
    processed_count = 0
    skipped_count = 0
    
    for structure_name in structures_to_process:
        if structure_name not in contours_dict:
            logger.warning(f"Structure '{structure_name}' not found. Available: {list(contours_dict.keys())}")
            skipped_count += 1
            continue
        
        logger.info(f"Processing structure: {structure_name}")
        contours_list = contours_dict[structure_name]
        contour_count = 0
        
        for contour_idx, contour_coords in enumerate(contours_list):
            try:
                # Convert world coordinates to voxel coordinates
                voxel_coords = []
                z_indices = set()
                
                for point_idx, (world_x, world_y, world_z) in enumerate(contour_coords):
                    try:
                        phys_point = (float(world_x), float(world_y), float(world_z))
                        voxel_idx = dicom_image.TransformPhysicalPointToContinuousIndex(phys_point)
                        voxel_coords.append(voxel_idx)
                        z_indices.add(int(round(voxel_idx[2])))
                    except Exception as e:
                        logger.debug(f"Error transforming point {point_idx}: {e}")
                        continue
                
                if not voxel_coords:
                    logger.debug(f"No valid coordinates for contour {contour_idx}")
                    continue
                
                # Convert to numpy array
                voxel_coords = np.array(voxel_coords)
                
                # Process each z-slice that has contour points
                for z_idx in z_indices:
                    z_idx = int(z_idx)
                    
                    # Bounds check
                    if z_idx < 0 or z_idx >= shape[2]:
                        logger.debug(f"Contour z-index {z_idx} out of bounds [0, {shape[2]-1}]")
                        continue
                    
                    try:
                        # Get x, y coordinates for this z-slice
                        coords_2d = voxel_coords[
                            np.abs(voxel_coords[:, 2] - z_idx) < 0.5
                        ]
                        
                        if len(coords_2d) < 3:  # Need at least 3 points for polygon
                            logger.debug(f"Insufficient points ({len(coords_2d)}) for z={z_idx}")
                            continue
                        
                        # Create binary mask for this slice
                        filled_poly = poly2mask(
                            coords_2d[:, 0], 
                            coords_2d[:, 1], 
                            [shape[0], shape[1]]
                        )
                        
                        # Use XOR to handle multiple contours
                        mask[z_idx, :, :] = np.logical_xor(
                            mask[z_idx, :, :], 
                            filled_poly
                        ).astype(np.uint8)
                        
                        contour_count += 1
                        
                    except Exception as e:
                        logger.debug(f"Error processing contour at z={z_idx}: {e}")
                        continue
            
            except Exception as e:
                logger.warning(f"Error processing contour {contour_idx} for {structure_name}: {e}")
                continue
        
        if contour_count > 0:
            logger.info(f"  ✓ {structure_name}: {contour_count} contours processed")
            processed_count += 1
        else:
            logger.warning(f"  ✗ {structure_name}: No contours could be processed")
            skipped_count += 1
    
    logger.info(f"Mask conversion complete: {processed_count} structures processed, {skipped_count} skipped")
    return mask


def convert_rtstruct_to_mask(rtstruct_path, dcm_image, output_path, 
                            structure_names=None):
    """
    Convert RT Struct to binary mask NIFTI.
    """
    
    if not os.path.exists(rtstruct_path):
        raise FileNotFoundError(f"RT Struct file not found: {rtstruct_path}")
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    logger.info(f"Loading RT Struct: {rtstruct_path}")
    contours_dict = extract_rtstruct_contours(rtstruct_path)
    
    if not contours_dict:
        raise ValueError("No contours found in RT Struct")
    
    logger.info(f"Found {len(contours_dict)} structures: {list(contours_dict.keys())}")
    
    logger.info("Converting contours to mask...")
    mask_array = contours_to_mask(contours_dict, dcm_image, structure_names)
    
    # Check if mask is empty
    if np.sum(mask_array) == 0:
        logger.warning("Resulting mask is empty (all zeros)")
    else:
        logger.info(f"Mask contains {np.sum(mask_array)} non-zero voxels")
    
    logger.info(f"Saving mask to: {output_path}")
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(dcm_image)
    
    sitk.WriteImage(mask_image, output_path)
    logger.info(f"✓ Mask saved: {output_path}")
    
    return mask_image


# ============================================================================
# DICOM SERIES TO 3D VOLUME CONVERSION
# ============================================================================

def sort_dicom_slices(dicom_files):
    """
    Sort DICOM files by slice location (z-position).
    """
    slices = []
    
    for fname in dicom_files:
        try:
            ds = pydicom.dcmread(fname, stop_before_pixels=True)
            z_pos = float(ds.ImagePositionPatient[2])
            slices.append((z_pos, fname))
        except Exception as e:
            logger.debug(f"Could not read position from {fname}: {e}")
            slices.append((0, fname))
    
    slices.sort(key=lambda x: x[0])
    return [fname for _, fname in slices]


def load_dicom_series(dcm_path, series_id=None):
    """
    Load DICOM series and convert to 3D SimpleITK image.
    """
    reader = sitk.ImageSeriesReader()
    
    logger.info(f"Loading DICOM series from: {dcm_path}")
    
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    
    if not dicom_names:
        raise ValueError(f"No DICOM files found in {dcm_path}")
    
    if series_id:
        logger.info(f"Filtering for Series InstanceUID: {series_id}")
        filtered_names = []
        for fname in dicom_names:
            ds = pydicom.dcmread(fname, stop_before_pixels=True)
            if ds.SeriesInstanceUID == series_id:
                filtered_names.append(fname)
        
        if filtered_names:
            dicom_names = filtered_names
        else:
            logger.warning(f"No files found for series {series_id}, using all files")
    
    logger.info(f"Found {len(dicom_names)} DICOM files, sorting by position...")
    dicom_names = sort_dicom_slices(dicom_names)
    
    reader.SetFileNames(dicom_names)
    dcm_image = reader.Execute()
    
    logger.info(f"✓ Loaded DICOM image - Shape: {dcm_image.GetSize()}, Spacing: {dcm_image.GetSpacing()}")
    
    return dcm_image


def convert_dicom_series_to_nifti(dcm_path, output_path, series_id=None):
    """
    Convert DICOM series to 3D NIFTI volume.
    """
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("CONVERTING DICOM SERIES TO 3D NIFTI")
    logger.info("=" * 60)
    
    dicom_image = load_dicom_series(dcm_path, series_id=series_id)
    
    logger.info(f"Saving to NIFTI: {output_path}")
    sitk.WriteImage(dicom_image, output_path)
    logger.info(f"✓ NIFTI volume saved: {output_path}")
    
    return dicom_image


def list_rtstruct_structures(rtstruct_path):
    """
    List all structures in an RT Struct file.
    """
    contours_dict = extract_rtstruct_contours(rtstruct_path)
    return list(contours_dict.keys())


# ============================================================================
# MAIN / CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert DICOM series to 3D NIFTI volume and RT Struct to mask NIFTI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert DICOM series to 3D volume
  python dicom_converter.py --dicom_dir /path/to/dicom --output /output/image.nii.gz
  
  # Convert RT Struct to mask (all structures)
  python dicom_converter.py --dicom_dir /path/to/dicom --rtstruct rt_struct.dcm --mask_output /output/mask.nii.gz
  
  # List available structures in RT Struct
  python dicom_converter.py --rtstruct rt_struct.dcm --list_structures
  
  # Debug RT Struct structure
  python dicom_converter.py --rtstruct rt_struct.dcm --debug_rtstruct
  
  # Convert specific structures only
  python dicom_converter.py --dicom_dir /path/to/dicom --rtstruct rt_struct.dcm --structures "Heart" "Lung" --mask_output /output/mask.nii.gz
  
  # Convert both DICOM and RT Struct
  python dicom_converter.py --dicom_dir /path/to/dicom --output /output/image.nii.gz --rtstruct rt_struct.dcm --mask_output /output/mask.nii.gz
        """
    )
    
    parser.add_argument('--dicom_dir', default=None,
                       help='Directory containing DICOM series')
    parser.add_argument('--output', default=None,
                       help='Output path for DICOM volume (.nii.gz)')
    parser.add_argument('--rtstruct', default=None,
                       help='Path to RT Struct DICOM file')
    parser.add_argument('--mask_output', default=None,
                       help='Output path for mask (.nii.gz)')
    parser.add_argument('--structures', nargs='+', default=None,
                       help='Specific structure names to convert')
    parser.add_argument('--list_structures', action='store_true',
                       help='List all structures in RT Struct file and exit')
    parser.add_argument('--debug_rtstruct', action='store_true',
                       help='Print RT Struct file structure and exit')
    parser.add_argument('--series_id', default=None,
                       help='Series Instance UID (for multi-series directories)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Debug RT Struct structure
    if args.debug_rtstruct:
        if not args.rtstruct:
            logger.error("--debug_rtstruct requires --rtstruct argument")
            return
        print_rtstruct_structure(args.rtstruct)
        return
    
    # List structures if requested
    if args.list_structures:
        if not args.rtstruct:
            logger.error("--list_structures requires --rtstruct argument")
            return
        logger.info(f"\n{'Available structures in RT Struct:':^60}")
        logger.info("-" * 60)
        try:
            structures = list_rtstruct_structures(args.rtstruct)
            if structures:
                for i, struct in enumerate(structures, 1):
                    print(f"  {i}. {struct}")
            else:
                print("  No structures found!")
        except Exception as e:
            logger.error(f"Error listing structures: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        return
    
    # Validate inputs
    if not args.dicom_dir and not args.rtstruct:
        parser.print_help()
        logger.error("\nError: Provide --dicom_dir and/or --rtstruct")
        return
    
    # Convert DICOM series to volume
    dicom_image = None
    if args.dicom_dir and args.output:
        try:
            dicom_image = convert_dicom_series_to_nifti(
                args.dicom_dir,
                args.output,
                series_id=args.series_id
            )
        except Exception as e:
            logger.error(f"Error converting DICOM series: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return
    elif args.dicom_dir:
        logger.warning("--dicom_dir provided but --output not specified. Skipping DICOM conversion.")
    
    # Convert RT Struct to mask
    if args.rtstruct and args.mask_output:
        logger.info("\n" + "=" * 60)
        logger.info("CONVERTING RT STRUCT TO MASK NIFTI")
        logger.info("=" * 60)
        
        try:
            # If DICOM image not loaded yet, load it
            if dicom_image is None:
                if not args.dicom_dir:
                    logger.error("RT Struct conversion requires DICOM image reference. Provide --dicom_dir")
                    return
                dicom_image = load_dicom_series(args.dicom_dir, series_id=args.series_id)
            
            convert_rtstruct_to_mask(
                args.rtstruct,
                dicom_image,
                args.mask_output,
                structure_names=args.structures
            )
        except Exception as e:
            logger.error(f"Error converting RT Struct: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return
    elif args.rtstruct:
        logger.warning("--rtstruct provided but --mask_output not specified. Skipping mask conversion.")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ CONVERSION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
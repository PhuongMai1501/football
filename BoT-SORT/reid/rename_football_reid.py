"""
Script để đổi tên các file ảnh trong folder football_reid
Quy tắc:
- Chia mỗi folder thành 2 phần: 50% đầu giữ nguyên c1, 50% sau đổi thành c2
- Giữ nguyên person_id và frame number
  
Ví dụ (nếu folder có 100 ảnh):
- 50 ảnh đầu: giữ nguyên c1
- 50 ảnh sau: đổi c1 -> c2
"""

import os
import re
from pathlib import Path


def parse_filename(filename):
    """
    Parse filename theo pattern: XXX_cY_fZZZZZZ
    Returns: (person_id, camera_id, frame_number) hoặc None nếu không match
    """
    # Pattern: 3 chữ số _ c + số _ f + nhiều chữ số
    pattern = r'^(\d{3})_c(\d+)_f(\d+)'
    match = re.match(pattern, filename)
    
    if match:
        person_id = match.group(1)
        camera_id = match.group(2)
        frame_str = match.group(3)
        frame_number = int(frame_str)
        return person_id, camera_id, frame_str, frame_number
    
    return None


def should_rename(file_index, total_files, camera_id):
    """
    Kiểm tra xem file có cần đổi tên không
    - Đổi tên 50% sau của các file (nửa sau)
    - Chỉ đổi nếu camera hiện tại là c1
    """
    half_point = total_files // 2
    return camera_id == '1' and file_index >= half_point


def generate_new_name(person_id, camera_id, frame_str, extension):
    """
    Tạo tên mới: đổi camera từ c1 thành c2
    """
    new_camera_id = '2'
    new_name = f"{person_id}_c{new_camera_id}_f{frame_str}{extension}"
    return new_name


def rename_files_in_directory(root_dir, dry_run=True):
    """
    Đổi tên các file trong tất cả các folder con của root_dir
    Quy tắc: 50% đầu giữ nguyên c1, 50% sau đổi thành c2
    
    Args:
        root_dir: Đường dẫn đến folder gốc (ví dụ: football_reid)
        dry_run: Nếu True, chỉ hiển thị những gì sẽ thay đổi mà không thực sự đổi tên
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"❌ Không tìm thấy folder: {root_dir}")
        return
    
    total_renamed = 0
    total_skipped = 0
    total_errors = 0
    
    print(f"📁 Đang quét folder: {root_dir}")
    print(f"{'🔍 Chế độ: DRY RUN (không thay đổi thực tế)' if dry_run else '✏️  Chế độ: RENAME (thay đổi thực tế)'}\n")
    
    # Duyệt qua các subfolder (train, gallery, query)
    for subfolder in ['train', 'gallery', 'query']:
        subfolder_path = root_path / subfolder
        if not subfolder_path.exists():
            continue
        
        print(f"\n{'='*60}")
        print(f"📂 Đang xử lý: {subfolder}")
        print(f"{'='*60}\n")
        
        # Duyệt qua từng person ID folder
        person_folders = sorted([d for d in subfolder_path.iterdir() if d.is_dir()])
        
        for person_folder in person_folders:
            # Lấy tất cả file trong folder này
            all_files = sorted([f for f in person_folder.iterdir() if f.is_file()])
            
            # Lọc các file match pattern
            valid_files = []
            for file_path in all_files:
                parsed = parse_filename(file_path.name)
                if parsed is not None:
                    valid_files.append((file_path, parsed))
            
            if len(valid_files) == 0:
                continue
            
            total_files = len(valid_files)
            half_point = total_files // 2
            
            print(f"📁 Folder: {subfolder}/{person_folder.name}")
            print(f"   📊 Tổng số file: {total_files}")
            print(f"   ✅ Giữ nguyên c1: {half_point} file đầu")
            print(f"   🔄 Đổi c1→c2: {total_files - half_point} file sau\n")
            
            renamed_count = 0
            skipped_count = 0
            
            # Xử lý từng file
            for idx, (file_path, parsed) in enumerate(valid_files):
                person_id, camera_id, frame_str, frame_number = parsed
                filename = file_path.name
                file_extension = file_path.suffix
                
                # Kiểm tra xem có cần đổi tên không
                if should_rename(idx, total_files, camera_id):
                    new_name = generate_new_name(person_id, camera_id, frame_str, file_extension)
                    new_path = file_path.parent / new_name
                    
                    if renamed_count < 5 or renamed_count >= total_files - half_point - 5:
                        # Chỉ in ra 5 file đầu và 5 file cuối để tránh spam
                        print(f"   📝 {filename} ➜ {new_name}")
                    elif renamed_count == 5:
                        print(f"   ... (đang đổi tên {total_files - half_point - 10} file khác) ...")
                    
                    if not dry_run:
                        try:
                            file_path.rename(new_path)
                            renamed_count += 1
                            total_renamed += 1
                        except Exception as e:
                            print(f"   ❌ Lỗi khi đổi {filename}: {e}")
                            total_errors += 1
                    else:
                        renamed_count += 1
                        total_renamed += 1
                else:
                    skipped_count += 1
                    total_skipped += 1
            
            print(f"   ✅ Hoàn thành folder {person_folder.name}: {renamed_count} đổi tên, {skipped_count} giữ nguyên\n")
    
    # In tổng kết
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT TOÀN BỘ:")
    print(f"   • Files sẽ đổi tên (c1→c2): {total_renamed}")
    print(f"   • Files giữ nguyên (c1): {total_skipped}")
    if total_errors > 0:
        print(f"   • Lỗi: {total_errors}")
    print("=" * 60)


def main():
    """
    Main function
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Đổi tên các file ảnh trong football_reid theo quy tắc frame chẵn/lẻ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  1. Xem trước những thay đổi (dry-run):
     python rename_football_reid.py
  
  2. Thực hiện đổi tên thực tế:
     python rename_football_reid.py --execute
  
  3. Chỉ định đường dẫn khác:
     python rename_football_reid.py --folder "D:/dataset/football_reid" --execute

Quy tắc đổi tên:
  - Trong mỗi folder person ID:
    + 50% file ĐẦU: GIỮ NGUYÊN c1
    + 50% file SAU: ĐỔI c1 → c2
        """
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        default=r'D:\test\train_reid\football_reid',
        help='Đường dẫn đến folder football_reid (mặc định: D:\\test\\train_reid\\football_reid)'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Thực hiện đổi tên thực tế (mặc định là dry-run)'
    )
    
    args = parser.parse_args()
    
    # Xác nhận với người dùng nếu là chế độ execute
    if args.execute:
        print("⚠️  CẢNH BÁO: Bạn đang chạy ở chế độ EXECUTE!")
        print("⚠️  Các file sẽ được đổi tên THỰC SỰ!")
        response = input("\n❓ Bạn có chắc chắn muốn tiếp tục? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Đã hủy thao tác.")
            return
        print()
    
    # Thực hiện đổi tên
    rename_files_in_directory(args.folder, dry_run=not args.execute)


if __name__ == '__main__':
    main()

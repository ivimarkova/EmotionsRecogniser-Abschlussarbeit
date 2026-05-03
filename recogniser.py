import sys
from realtime_mode import run_realtime
from image_mode import run_image_mode
 
 
def print_menu():
    print("\n" + "="*50)
    print("   EMOTION RECOGNITION SYSTEM")
    print("="*50)
    print("  [1] Real-Time Webcam Analysis")
    print("  [2] Image-Based Analysis")
    print("  [Q] Quit")
    print("="*50)
 
 
def main():
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip().lower()
 
        if choice == '1':
            run_realtime()
        elif choice == '2':
            run_image_mode()
        elif choice == 'q':
            print("\nGoodbye!\n")
            sys.exit(0)
        else:
            print("[!] Invalid choice. Please enter 1, 2, or Q.")
 
 
if __name__ == "__main__":
    main()
 
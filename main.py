import sys
from app.views.main_view import MainView

def main():
    try:
        app = MainView()
        app.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted from keyboard.")
    except Exception as e:
        print(f"Error in application: {e}")
    finally:
        # Clean up and close properly
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        sys.exit(0)

if __name__ == "__main__":
    main()
import os
import sys 
import platform
cwd=os.getcwd()
val=os.path.isdir(f"C:\\Users\\Karthik\\.vscode\\aiml\\advanced_sklearn")
url=f"C:\\Users\\Karthik\\.vscode\\aiml\\advanced_sklearn"
# print(cwd)
# if val:
#     os.mkdir("sample")
# if val:
#     os.rmdir("sample")
# print("directory : ",os.path.dirname(url))
# print(f"\n5.4 System information:")
# print(f"OS name: {os.name}")  
# print(f"Platform: {sys.platform}")
# print(f"System: {platform.system()}")
# print(f"Release: {platform.release()}")
# print(f"Machine: {platform.machine()}")

# print(f"\n5.5 Process information:")
# print(f"Process ID (PID): {os.getpid()}")
# print(f"Parent PID: {os.getppid()}")



     

# def list_all_paths(root):
#     for dirpath, dirnames, filenames in os.walk(root):
#         print(dirpath) 
#         for name in filenames:
#             print(os.path.join(dirpath, name))

# list_all_paths("C:\\") 

def all_dir(root):
    for dirpath,dirnames,filenames in os.walk(root):
        print(dirpath)
        for file in filenames:
            if file.endswith(".py"):
                 print(file)
all_dir(f"C:\\Users\\Karthik\\.vscode\\aiml\\advanced_sklearn")
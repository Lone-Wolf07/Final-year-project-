import tkinter as tk
from tkinter import filedialog

def open_file():
    path = filedialog.askopenfilename()
    if path:
        with open(path, "r") as file:
            text_area.delete("1.0", tk.END)
            text_area.insert(tk.END, file.read())

def save_file():
    path = filedialog.asksaveasfilename(defaultextension=".txt")
    if path:
        with open(path, "w") as file:
            file.write(text_area.get("1.0", tk.END))

root = tk.Tk()
root.title("Text Editor")

text_area = tk.Text(root)
text_area.pack(fill=tk.BOTH, expand=True)

menu_bar = tk.Menu(root)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open", command=open_file)
file_menu.add_command(label="Save", command=save_file)
menu_bar.add_cascade(label="File", menu=file_menu)
root.config(menu=menu_bar)

root.mainloop()

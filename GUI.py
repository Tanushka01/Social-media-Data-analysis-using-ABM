import tkinter as tk
from plotly_test import *

root = tk.Tk()
root.geometry("500x500")  # set the size of the window to 500x500
root.title("Choose Testing Parameters")
root.configure(bg="lightblue")  # set the background color to red


# Create a label widget
label = tk.Label(root, text="Enter your name:", bg="lightblue")
label.pack(padx=2, pady=2)

# Create an entry widget
entry = tk.Entry(root, bg="pink", borderwidth=0, highlightthickness=0)

entry.pack(padx=2, pady=2)


def submit():
    name = entry.get()
    make_plot(int(name),,


# Create a button widget
button = tk.Button(root, text="Submit", command=submit, bg="lightblue", activebackground="gray")
button.pack(padx=2, pady=2)

root.mainloop()

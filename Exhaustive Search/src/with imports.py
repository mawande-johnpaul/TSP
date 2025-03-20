import tkinter as tk
import json

app = tk.Tk()

WIDTH = app.winfo_screenwidth()
HEIGHT = app.winfo_screenheight()

app.geometry(f'{WIDTH}x{HEIGHT}')
app.title('Notes')

users_file = 'Users.json'
try:
    with open(users_file, 'w+') as file:
        users = json.load(file)

except:
    users=dict()

class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.notes = notebook.get_notes(self.username)


class Note:
    def __int__(self, name, content, filename='', created='', edited='', author=''):
        self.name = name
        self.content = content
        self.filename = filename
        self.created = created
        self.edited = edited
        self.author = author

class Notebook:
    def __init__(self, users_dictionary):
        self.users = users_dictionary

    def get_notes(self, username):
        return self.users[username]['notes']

    def create_note(self):
        clear_frame(content_section)

        name_input = tk.Entry(content_section, width=int(0.98*0.77*WIDTH))
        name_input.place(x=0.1*0.77*WIDTH, y=10)

def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

notebook = Notebook(users)

# frames
sidebar = tk.Frame(app, width=int(0.195*WIDTH), height=int(0.935*HEIGHT), background='gray')
sidebar.place(x=10, y=10)

content_section = tk.Frame(app, width=int(0.770*WIDTH), height=int(0.935*HEIGHT), background='gray')
content_section.place(x=int(0.215*WIDTH), y=int(0.0095*HEIGHT))

#buttons
new_note = tk.Button(sidebar, width=int(0.15*0.195*WIDTH), height=1, command=notebook.create_note)
new_note.place(x=10, y=10)

app.mainloop()








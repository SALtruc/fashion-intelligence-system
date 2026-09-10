"""Local fashion catalogue tagging demo. Run: python task1_demo.py"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import ImageOps, ImageTk
import torch
from src.task1_inference import Predictor, DEFAULT_ARTIFACTS, load_image

# Catalogue contact-sheet palette; the product photograph is the focal point.
COLORS = {'paper':'#F2F5F8', 'ink':'#24384A', 'blue':'#245F96',
          'muted':'#586C7C', 'line':'#CED9E3', 'white':'#FFFFFF'}


class CatalogueApp:
    def __init__(self, root, artifacts=DEFAULT_ARTIFACTS):
        self.root = root
        self.artifacts = artifacts
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.predictor = None
        self.future = None
        self.path = None
        self.result = None
        self.photo = None
        self.closed = False
        root.title('Catalogue studio | Fashion item tagging')
        root.geometry('940x730')
        root.minsize(760, 680)
        root.configure(bg=COLORS['paper'])
        root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI',11), padding=(16,10))
        style.configure('Primary.TButton', background=COLORS['blue'], foreground='white')
        style.map('Primary.TButton', background=[('active','#194D7D'),('disabled',COLORS['line'])])
        style.configure('Treeview', rowheight=32, font=('Segoe UI',11), background='white', fieldbackground='white')
        style.configure('Treeview.Heading', font=('Segoe UI',10,'bold'))
        style.configure('TCombobox', padding=6, font=('Segoe UI',11))
        outer = tk.Frame(root, bg=COLORS['paper'], padx=30, pady=24)
        outer.pack(fill='both', expand=True)
        self.label(outer, 'CATALOGUE STUDIO', 10, color=COLORS['blue']).pack(anchor='w')
        self.label(outer, 'Give your product a label.', 27, family='Georgia').pack(anchor='w', pady=(6,8))
        self.label(outer, 'Choose a product photo, review the suggestions, then save your label.',11).pack(anchor='w')
        body = tk.Frame(outer, bg=COLORS['paper'])
        body.pack(fill='both', expand=True, pady=22)
        body.columnconfigure(0, weight=1, uniform='panel')
        body.columnconfigure(1, weight=1, uniform='panel')
        body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg='white', padx=18, pady=18, highlightbackground=COLORS['line'], highlightthickness=1)
        left.grid(row=0,column=0,sticky='nsew',padx=(0,12))
        self.label(left, 'PRODUCT PHOTO',10,bg='white').pack(anchor='w')
        self.preview = tk.Label(left,text='One item. A clear background.\n\nChoose a JPG or PNG to begin.',bg='white',fg=COLORS['muted'],font=('Segoe UI',12))
        self.preview.pack(fill='both',expand=True,pady=12)
        self.filename = self.label(left,'No photo selected',10,bg='white')
        self.filename.pack(fill='x',pady=(0,10))
        self.choose = ttk.Button(left,text='Choose photo',style='Primary.TButton',command=self.select)
        self.choose.pack(fill='x')
        right = tk.Frame(body,bg=COLORS['paper'],padx=8)
        right.grid(row=0,column=1,sticky='nsew')
        self.label(right,'SUGGESTED ARTICLE TYPES',10,color=COLORS['blue']).pack(anchor='w')
        self.headline = self.label(right,'Ready when you are',20,family='Georgia')
        self.headline.pack(anchor='w',pady=(10,14))
        self.ranking = ttk.Treeview(right,columns=('label','score'),show='headings',height=5,selectmode='browse')
        self.ranking.heading('label',text='Article type');self.ranking.heading('score',text='Model score')
        self.ranking.column('label',width=210,stretch=True);self.ranking.column('score',width=100,anchor='e',stretch=False)
        self.ranking.pack(fill='x')
        self.ranking.bind('<<TreeviewSelect>>',self.use_suggestion)
        self.label(right,'Scores rank suggestions; they are not verified probabilities of being correct.',10,wrap=330).pack(anchor='w',pady=(8,20))
        self.label(right,'Your reviewed label',11).pack(anchor='w')
        self.review = ttk.Combobox(right,state='disabled')
        self.review.pack(fill='x',pady=(5,10))
        self.save = ttk.Button(right,text='Save reviewed label',command=self.save_label,state='disabled')
        self.save.pack(fill='x')
        self.status = self.label(outer,'Photos are processed on this computer. No upload or account is needed.',10,wrap=800)
        self.status.pack(anchor='w')
        self.label(outer,'Catalogue-photo demo • Check the label before using it. Unfamiliar items may still receive a suggestion.',10,wrap=800).pack(anchor='w',pady=(8,0))
        root.bind('<Control-o>',lambda _: self.select() if self.future is None else None)
        root.after(100,self.poll)

    def label(self,parent,text,size,color=None,family='Segoe UI',bg=None,wrap=0):
        return tk.Label(parent,text=text,font=(family,size),fg=color or COLORS['ink'],
                        bg=bg or COLORS['paper'],anchor='w',justify='left',wraplength=wrap)

    def select(self):
        path = filedialog.askopenfilename(title='Choose a product photo',filetypes=[('Product photos','*.jpg *.jpeg *.png'),('All files','*.*')])
        if path:
            self.classify_path(Path(path))

    def classify_path(self,path):
        if self.future is not None:
            return
        self.result = None
        self.path = path
        self.ranking.delete(*self.ranking.get_children())
        self.review.set('');self.review.configure(state='disabled')
        self.save.configure(state='disabled');self.choose.configure(state='disabled')
        self.preview.configure(image='',text='Reading product photo…')
        self.headline.configure(text='Finding suggestions…')
        self.filename.configure(text=path.name[:45])
        self.status.configure(text='Preparing the model and photo…')
        self.future = self.executor.submit(self.classify,path)

    def classify(self,path):
        image = load_image(path)
        if self.predictor is None:
            self.predictor = Predictor(self.artifacts,device='cpu')
        result = self.predictor.predict(image)
        return image,result

    def poll(self):
        if self.closed:
            return
        if self.future is not None and self.future.done():
            try:
                image,self.result = self.future.result()
                preview = ImageOps.contain(image,(320,340))
                self.photo = ImageTk.PhotoImage(preview,master=self.root)
                self.preview.configure(image=self.photo,text='')
                self.headline.configure(text=self.result['articleType'],wraplength=330)
                for row in self.result['suggestions']:
                    self.ranking.insert('', 'end',values=(row['label'],f"{row['score']:.3f}"))
                self.review.configure(values=self.predictor.classes,state='readonly')
                self.review.set(self.result['articleType'])
                self.save.configure(state='normal')
                self.status.configure(text=f"Suggestion ready in {self.result['seconds']:.2f} s after model loading. Review or change the label before saving.")
            except Exception as exc:
                self.preview.configure(image='',text='This photo could not be processed.\nChoose another JPG or PNG.')
                self.headline.configure(text='Could not classify')
                self.status.configure(text=str(exc),wraplength=800)
            finally:
                self.future = None
                self.choose.configure(state='normal')
        self.root.after(100,self.poll)

    def use_suggestion(self,event=None):
        selected = self.ranking.selection()
        if selected and self.result:
            self.review.set(self.ranking.item(selected[0],'values')[0])

    def save_label(self):
        if self.result is None:
            return
        destination = filedialog.asksaveasfilename(title='Save reviewed label',defaultextension='.csv',initialfile='reviewed_label.csv',filetypes=[('CSV','*.csv')])
        if not destination:
            return
        try:
            self.export_review(Path(destination))
            self.status.configure(text='Reviewed label saved. Choose another photo to continue.')
        except OSError as exc:
            messagebox.showerror('Could not save label',str(exc))

    def export_review(self,destination):
        if self.result is None or self.review.get() not in self.predictor.classes:
            raise ValueError('Choose a valid reviewed label first.')
        row = {'image':self.path.name,'suggested_articleType':self.result['articleType'],
               'reviewed_articleType':self.review.get(),'model':self.result['arm']}
        # Spreadsheet applications interpret certain prefixes as formulas.
        row = {key: "'"+value if value.startswith(('=','+','-','@')) else value for key,value in row.items()}
        with destination.open('w',encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=False,cancel_futures=True)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = tk.Tk()
    CatalogueApp(root,args.artifacts)
    root.mainloop()


if __name__ == '__main__':
    main()

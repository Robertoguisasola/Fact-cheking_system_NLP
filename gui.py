import tkinter as tk
from tkinter import scrolledtext
from model import verificar_afirmacion

def run_verification():
    claim = input_text.get("1.0", tk.END).strip()
    if not claim:
        return
    output_text.config(state='normal')
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "⏳ Verificando, espera por favor...\n")
    output_text.update()
    result = verificar_afirmacion(claim)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, result)
    output_text.config(state='disabled')

# 🧪 GUI
root = tk.Tk()
root.title("Fact-checking system")
root.geometry("800x600")
root.minsize(600, 500)
root.configure(padx=20, pady=20)

tk.Label(root, text="Introduce la afirmación a verificar:", font=("Arial", 14)).pack(anchor="w", pady=(0, 10))
input_text = scrolledtext.ScrolledText(root, height=5, width=80, font=("Arial", 12), wrap=tk.WORD)
input_text.pack(fill="both", expand=False, padx=5)

tk.Button(root, text="Verificar Afirmación", font=("Arial", 12), command=run_verification).pack(pady=15)
tk.Label(root, text="Respuesta del sistema:", font=("Arial", 14)).pack(anchor="w", pady=(10, 5))

output_text = scrolledtext.ScrolledText(root, height=15, width=80, font=("Courier", 11), wrap=tk.WORD)
output_text.pack(fill="both", expand=True, padx=5)
output_text.config(state='disabled')

root.mainloop()
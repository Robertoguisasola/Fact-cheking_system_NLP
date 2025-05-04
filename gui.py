import tkinter as tk
from tkinter import scrolledtext
from model import verificar_afirmacion

def run_verification():
    """
    Executes the claim verification pipeline in a Tkinter GUI.

    Retrieves the user's input from the input text widget, verifies the claim using
    an external function `verificar_afirmacion`, and displays the result in the output text widget.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Get the user's input claim from the text box
    claim = input_text.get("1.0", tk.END).strip()
    if not claim:
        return

    # Prepare the output box for new content
    output_text.config(state='normal')  # Enable editing the answer box
    output_text.delete("1.0", tk.END)  # Clear any existing text to modify it
    output_text.insert(tk.END, "⏳ Verifying, please wait...\n")
    output_text.update()  # Force UI update

    # Call the core fact-checking function with the user's claim
    result = verificar_afirmacion(claim)

    # Display the result in the output text box
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, result)  # Show the final result
    output_text.config(state='disabled')  # Disable editing again, to avoid problems

# Initialize the main application window
root = tk.Tk()
root.title("Fact-checking system")
root.geometry("800x600")
root.minsize(600, 500)
root.configure(padx=20, pady=20) # Padding around the content

# Input label and text area for the user to type a claim
tk.Label(root, text="Introduce the claim to be verified:", font=("Arial", 14)).pack(anchor="w", pady=(0, 10))
input_text = scrolledtext.ScrolledText(root, height=5, width=80, font=("Arial", 12), wrap=tk.WORD)
input_text.pack(fill="both", expand=False, padx=5)

# Button to trigger the verification logic
tk.Button(root, text="Verify claim", font=("Arial", 12), command=run_verification).pack(pady=15)

# Output label and area to display the verification result
tk.Label(root, text="System answer", font=("Arial", 14)).pack(anchor="w", pady=(10, 5))
output_text = scrolledtext.ScrolledText(root, height=15, width=80, font=("Courier", 11), wrap=tk.WORD)
output_text.pack(fill="both", expand=True, padx=5)
output_text.config(state='disabled') # Make output read-only 

root.mainloop()
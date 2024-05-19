import tkinter as tk
from tkinter import messagebox
import requests

def predict():
    target_value = target_entry.get()
    rain_value = rain_entry.get()
    time_value = time_entry.get()

    # Preverjanje, ali so vneseni podatki pravilni
    if not target_value or not rain_value or not time_value:
        messagebox.showerror("Napaka", "Prosim, vnesite vse podatke.")
        return

    # Pošiljanje zahtevka na Flask aplikacijo
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json={"target": target_value, "rain": rain_value, "time": time_value})
        prediction = response.json()["prediction"]
        messagebox.showinfo("Napoved", f"Napovedano število prostih mest za kolesa: {prediction}")
    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri pošiljanju zahtevka: {e}")

# Ustvarjanje glavnega okna
root = tk.Tk()
root.title("Napovedovanje števila prostih mest za kolesa")

# Dodajanje vnosnih polj
tk.Label(root, text="Ciljna vrednost:").pack()
target_entry = tk.Entry(root)
target_entry.pack()

tk.Label(root, text="Padavine (da/ne):").pack()
rain_entry = tk.Entry(root)
rain_entry.pack()

tk.Label(root, text="Čas (HH:MM):").pack()
time_entry = tk.Entry(root)
time_entry.pack()

# Gumb za napoved
predict_button = tk.Button(root, text="Napovej", command=predict)
predict_button.pack()

# Začetek zanke za prikazovanje okna
root.mainloop()

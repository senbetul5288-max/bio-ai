# Bitki bilgisi analizi

bitki = {
    "isim": "Lavanta",
    "renk": "Mor",
    "koku": "Hoş ve aromatik",
    "su_ihtiyaci": "Az",
    "sicaklik": "Ilıman iklim",
    "boy": 40,  # cm
}

def ozet_yaz(bitki):
    print("Bitki Adı:", bitki["isim"])
    print("Renk:", bitki["renk"])
    print("Koku:", bitki["koku"])
    print("Sulama:", bitki["su_ihtiyaci"])
    print("Büyüme Boyu:", bitki["boy"], "cm")

ozet_yaz(bitki)

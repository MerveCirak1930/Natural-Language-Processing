


def bayes_theorem(prior_A, prob_B_given_A, prob_B_given_not_A):
    # B olayının toplam olasılığı
    prob_B = (prob_B_given_A * prior_A) + (prob_B_given_not_A * (1 - prior_A))
    
    # Bayes Teoremi'ne göre güncellenmiş olasılık (posterior)
    posterior_A_given_B = (prob_B_given_A * prior_A) / prob_B
    return posterior_A_given_B



use_default = input("\nVarsayılan olasılıklar kullanılacak! Olasılıkları değiştirmek için ? e(evet)e tıklayınız:")

    
if use_default == "e":

    # Kullanıcıdan giriş alma
    prior_Hastalık = float(input("Hastalığın toplumda görülme oranını girin (0-1 arası): "))
    prob_Pozitif_given_Hastalık = float(input("Hasta olanlarda testin pozitif çıkma olasılığını girin (0-1 arası): "))
    prob_Pozitif_given_Saglikli = float(input("Sağlıklı kişilerde testin pozitif çıkma olasılığını girin (0-1 arası): "))

else:
    # Öncül olasılıklar
    prior_Hastalık = 0.01  # Hastalığın toplumda görülme oranı
    prob_Pozitif_given_Hastalık = 0.9  # Hasta olanlarda testin pozitif çıkma olasılığı
    prob_Pozitif_given_Saglikli = 0.05  # Sağlıklı kişilerde testin pozitif çıkma olasılığı



# Bayes Teoremi uygulama
sonuc = bayes_theorem(prior_Hastalık, prob_Pozitif_given_Hastalık, prob_Pozitif_given_Saglikli)
print(f"Test sonucu pozitif olduğunda gerçekten hasta olma olasılığı: {sonuc:.2%}")


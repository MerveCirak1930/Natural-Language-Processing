import pandas as pd
import os

# Genişletilmiş Türkçe e-posta veri seti
data = {
    "text": [
        "Bedava tatil kazandınız, hemen tıklayın!",
        "Fatura ödemeniz gecikti, lütfen ödeme yapın.",
        "Büyük indirim fırsatlarını kaçırmayın!",
        "Banka hesap özetiniz ektedir.",
        "Şanslı çekilişe katılın, ödüller kazanın!",
        "Vergi borcunuz bulunmaktadır, detaylar için tıklayın.",
        "Bugünkü toplantınız saat 14:00'te başlayacak.",
        "Kredi kartı borcunuz ödenmedi, işlem yapmanız gerekiyor!",
        "Para kazanın! Sadece birkaç adımda zengin olabilirsiniz!",
        "Öğrenci indirimlerinden faydalanmak için kayıt olun!",
        "Telefon faturanızı son ödeme tarihine kadar ödeyiniz.",
        "Kazandınız! 1000 TL değerinde hediye çeki sizi bekliyor!",
        "Üyeliğinizin süresi dolmak üzere, yenilemek için tıklayın.",
        "Havale işleminiz başarıyla tamamlandı.",
        "Size özel kredi fırsatlarını kaçırmayın!",
        "Hesabınıza yetkisiz erişim tespit edildi, hemen kontrol edin!",
        "Bu haftaya özel büyük indirimler! Kaçırmayın!",
        "Havale talebiniz işleme alınmıştır.",
        "Kargonuz yola çıktı, takip etmek için tıklayın.",
        "Kredi kartı limitinizi artırmak için başvuru yapabilirsiniz.",
        "Seçili ürünlerde %50'ye varan indirimler başladı!",
        "Havale işlemi için hesap bilgilerinizi güncelleyin!",
        "VIP üyelik fırsatlarını kaçırmayın, hemen üye olun!",
        "Hesap bakiyeniz düşük, lütfen yükleme yapınız.",
        "Kullanıcı hesabınız askıya alındı, tekrar aktif etmek için tıklayın!"
    ],
    "label": [
        1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
        0, 1, 1, 0, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 0, 1
    ]  # 1 = Spam, 0 = Normal
}

# Veri setini pandas DataFrame olarak oluşturma
df = pd.DataFrame(data)

# Kaydetme yolu
save_path = "2-Basic Statistics_and_Bayesian Models/turkish_spam_dataset.csv"


# Klasör var mı kontrol et, yoksa oluştur
os.makedirs(os.path.dirname(save_path), exist_ok=True)
  
# CSV olarak kaydetme
df.to_csv(save_path, index=False, encoding='utf-8')

print(f"Veri seti '{save_path}' konumuna kaydedildi.")


def sifir_koruma(func):
    def wrapper(*args,**kwargs):
        if args[1]==0:
            raise ValueError("Bir sayı sıfıra bölünemez !!!")
        else:
             return func(*args, **kwargs)
        
    return wrapper


@sifir_koruma
def fonksiyon(sayi_1,sayi_2):
    sonuc= sayi_1/sayi_2
    print(sonuc)


fonksiyon(10,0)
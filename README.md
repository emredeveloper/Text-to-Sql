# Text-to-SQL Demos

Bu depo, doğal dil ile SQL sorguları üretmek ve çalıştırmak için hazırlanmış iki farklı Streamlit arayüzü içerir:

- **`app.py`**: Yerel bir Llama.cpp modelini kullanarak metni SQL'e dönüştürür.
- **`text_to_sql_ollama.py`**: Ollama servisindeki modellerden yararlanır.

Her iki arayüz de örnek olarak sağlanan `example.db` SQLite veritabanıyla çalışır.

## Ön Koşullar

- Python 3.10 veya üzeri
- (Ollama arayüzü için) [Ollama](https://ollama.com) kurulu ve `ollama serve` komutuyla çalışan bir servis

## Kurulum

Depoyu klonladıktan sonra bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

> Not: `requirements.txt` dosyasındaki sürüm aralıkları, LangChain ekosistemindeki hızlı değişikliklerden kaynaklanabilecek bozulmaları azaltmak için belirlenmiştir. Farklı bir Python ortamında çalışıyorsanız sanal ortam kullanmanız tavsiye edilir.

## Llama.cpp Arayüzünü Çalıştırmak (`app.py`)

1. Hugging Face üzerindeki modeli indirin. Uygulama ilk çalıştırmada dosya mevcut değilse indirmeyi deneyecektir. Dilerseniz aşağıdaki komutu manuel olarak çalıştırabilirsiniz:

   ```bash
   curl -L -o phi-3-sql.Q4_K_M.gguf \
     https://huggingface.co/omeryentur/phi-3-sql/resolve/main/phi-3-sql.Q4_K_M.gguf
   ```

2. Dosyanın bütünlüğünü kontrol etmek için MD5 karmasını hesaplayıp `app.py` içerisindeki `expected_md5` alanına ekleyebilirsiniz:

   ```bash
   md5sum phi-3-sql.Q4_K_M.gguf
   ```

3. Streamlit uygulamasını başlatın:

   ```bash
   streamlit run app.py
   ```

4. Tarayıcıda açılan arayüzde doğal dilde sorgu girerek sonuçları görüntüleyebilirsiniz.

## Ollama Arayüzünü Çalıştırmak (`text_to_sql_ollama.py`)

1. Ollama servisinizde kullanılacak modeli indirin. Örneğin `gemma3` modeli için:

   ```bash
   ollama pull gemma3
   ```

2. Streamlit uygulamasını başlatın:

   ```bash
   streamlit run text_to_sql_ollama.py
   ```

3. Sol taraftaki panelden kullanılacak modeli seçebilir, veritabanı ihracı yapabilir veya sorgu geçmişini temizleyebilirsiniz.

## Güvenlik ve En İyi Uygulamalar

- Uygulamalardaki "Add Data" bölümü yalnızca tek bir `INSERT` ifadesinin çalıştırılmasına izin verir. Bu sınırlama kötü niyetli SQL enjeksiyonlarının önüne geçmeye yardımcı olur.
- Üretilen SQL'i çalıştırmadan önce gözden geçirmeniz önerilir. Karmaşık sorgular için `sqlparse` gibi araçlarla ek doğrulama yapılabilir.

## Testler

Projede temel yardımcı fonksiyonları doğrulayan pytest testleri bulunur. Testleri çalıştırmak için:

```bash
pytest
```

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında sunulmaktadır.

import requests
import pandas as pd
from telegram import Bot
import asyncio
import numpy as np
from datetime import datetime, timedelta
from telegram.ext import Application, CommandHandler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
bot_token = 'BOT_TOKEN'
chat_id = 'CHAT_ID'
subscribed_users = set()


bot = Bot(token=bot_token)

exclude_stablecoins = ['FUSD/USDT', 'WBTC/USDT', 'WETH/USDT', 'USDT/USDT', 'BUSD/USDT', 'DAI/USDT',
                       'TUSD/USDT', 'PAX/USDT', 'PAXG/USDT', 'EUR/USDT', 'TCT/USDT', 'USDS/USDT', 'VEN/USDT',
                       'USDC/USDT',
                       'BCHABC/USDT', 'SNT/USDT', 'SLP/USDT', 'STP/USDT', 'STRAX/USDT', 'GLMR/USDT', 'FRONT/USDT',
                       'MDT/USDT',
                       'RAD/USDT', 'IDEX/USDT', 'AGIX/USDT', 'CVX/USDT', 'FTT/USDT', 'RAY/USDT', 'DGB/USDT', 'SC/USDT',
                       'CTK/USDT', 'CVC/USDT', 'OCEAN/USDT', 'WAVES/USDT', 'STPT/USDT']

coin_data_cache = {}

CACHE_EXPIRY_DURATION = timedelta(minutes=15)
async def start(update, context):
    user_id = update.effective_user.id
    subscribed_users.add(user_id)
    await update.message.reply_text('You will now receive trading signals!')

async def send_to_subscribers(message):
    for user_id in subscribed_users:
        print(user_id)
        try:
            await bot.send_message(chat_id=user_id, text=message)
        except Exception as e:
            print(f"Failed to send to {user_id}: {e}")

def get_binance_futures_pairs():
    url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
    try:
        response = requests.get(url)
        data = response.json()
        futures_pairs = [symbol['symbol'] for symbol in data['symbols'] if
                         symbol['quoteAsset'] == 'USDT' and symbol['contractType'] == 'PERPETUAL']
        print(f"Vadeli işlem çiftleri alındı: {futures_pairs}")
        return futures_pairs
    except Exception as e:
        print(f"Vadeli işlem çiftleri alınırken hata oluştu: {e}")
        return []

def fetch_futures_data_multiple_intervals(symbol):
    intervals = ['15m','30m', '1h']
    data = {}
    for interval in intervals:
        data[interval] = fetch_futures_data(symbol, interval)
    return data
def fetch_futures_data(symbol, interval='15m'):
    print(f"Veri çekiliyor: {symbol}, interval: {interval}")
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=100'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


# Teknik analiz indikatör fonksiyonları
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, short_period=12, long_period=26, signal_period=9):
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal


def compute_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = (high.diff(1).where((high.diff(1) > low.diff(1)) & (high.diff(1) > 0), 0))
    minus_dm = (low.diff(1).where((low.diff(1) > high.diff(1)) & (low.diff(1) > 0), 0))

    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx


def compute_cci(df, period=20):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = (typical_price - sma).rolling(window=period).mean()
    return (typical_price - sma) / (0.015 * mad)



def compute_bollinger_bands(df, window=20, std_dev=2):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, lower_band

def compute_stochastic_rsi(df, period=14):
    min_price = df['close'].rolling(window=period).min()
    max_price = df['close'].rolling(window=period).max()
    return 100 * (df['close'] - min_price) / (max_price - min_price)

def compute_fibonacci_retracement(df):
    max_price = df['high'].max()
    min_price = df['low'].min()
    levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    retracement_levels = [(max_price - (max_price - min_price) * level) for level in levels]
    return retracement_levels

def compute_obv(df):
    return (np.sign(df['close'].diff()) * df['volume']).cumsum()
def compute_vwap(df):
    cumulative_volume = df['volume'].cumsum()
    return (df['close'] * df['volume']).cumsum() / cumulative_volume
def find_support_resistance(df):
    support = df['low'].min()
    resistance = df['high'].max()
    return support, resistance

def calculate_take_profit_and_stop_loss(entry_price, long=True):
    if long:
        take_1 = entry_price * 1.015
        take_2 = entry_price * 1.03
        take_3 = entry_price * 1.045
        take_4 = entry_price * 1.06
        stop_loss = entry_price * 0.97
    else:
        take_1 = entry_price * 0.985
        take_2 = entry_price * 0.97
        take_3 = entry_price * 0.965
        take_4 = entry_price * 0.94
        stop_loss = entry_price * 1.03
    return take_1, take_2, take_3, take_4, stop_loss


# Haber Duyarlılık Analizi (Sentiment Analysis)
def analyze_market_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']


# Yüzde fark hesaplama
def calculate_proximity(current_price, level):
    return abs(current_price - level) / level * 100


# Risk/Ödül Oranı Hesaplama
def compute_risk_reward(entry_price, stop_loss, take_profit):
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    return reward / risk


def obv_analysis(df):
    obv = compute_obv(df)  # OBV serisini alıyoruz

    # OBV serisinin son iki değerini karşılaştırıyoruz
    if obv.iloc[-1] > obv.iloc[-2]:  # Son değer bir önceki değerden büyükse hacim artışı var
        return "buy_signal"
    elif obv.iloc[-1] < obv.iloc[-2]:  # Son değer bir önceki değerden küçükse hacim düşüşü var
        return "sell_signal"
    return "neutral"


def vwap_analysis(df):
    vwap = compute_vwap(df).iloc[-1]
    close_price = df['close'].iloc[-1]

    if close_price > vwap:
        return "buy_signal"  # Fiyat VWAP'in üstünde
    elif close_price < vwap:
        return "sell_signal"  # Fiyat VWAP'in altında
    return "neutral"


def fibonacci_analysis(df):
    fib_levels = compute_fibonacci_retracement(df)
    current_price = df['close'].iloc[-1]

    for level in fib_levels:
        if abs(current_price - level) / level < 0.01:  # %1'lik sapma içinde
            if current_price < level:
                return "buy_signal"  # Fiyat fib seviyesinin altında
            else:
                return "sell_signal"  # Fiyat fib seviyesinin üstünde
    return "neutral"


def stochastic_rsi_analysis(df):
    stoch_rsi = compute_stochastic_rsi(df).iloc[-1]
    if stoch_rsi < 20:
        return "buy_signal"  # Aşırı satım
    elif stoch_rsi > 80:
        return "sell_signal"  # Aşırı alım
    return "neutral"


def bollinger_analysis(df):
    upper_band, lower_band = compute_bollinger_bands(df)
    close_price = df['close'].iloc[-1]

    if close_price > upper_band.iloc[-1]:
        return "sell_signal"  # Aşırı alım
    elif close_price < lower_band.iloc[-1]:
        return "buy_signal"  # Aşırı satım
    return "neutral"


def adx_analysis(df):
    adx = compute_adx(df).iloc[-1]
    if adx > 25:
        return "strong_trend"  # Güçlü trend
    elif adx < 20:
        return "weak_trend"  # Zayıf trend
    return "neutral"


def macd_analysis(df):
    macd, signal = compute_macd(df['close'])
    macd_last = macd.iloc[-1]
    signal_last = signal.iloc[-1]

    if macd_last > signal_last:
        return "buy_signal"  # MACD, sinyal çizgisinin üstünde; trend yukarı
    elif macd_last < signal_last:
        return "sell_signal"  # MACD, sinyal çizgisinin altında; trend aşağı
    return "neutral"


def rsi_analysis(df):
    rsi = compute_rsi(df['close'], 14).iloc[-1]
    if rsi < 30:
        return "buy_signal"  # Aşırı satım
    elif rsi > 70:
        return "sell_signal"  # Aşırı alım
    return "neutral"


def is_cached(symbol):
    if symbol in coin_data_cache:
        # Cache'deki süresi dolmuşsa sil ve False döndür
        if datetime.now() > coin_data_cache[symbol]['expiry']:
            del coin_data_cache[symbol]
            return False
        return True
    return False


def cache_coin_data(symbol, entry_price, take_1, take_2, take_3, take_4, stop_loss, long=True, investment=1000):
    # Cache'e coin verilerini kaydet
    coin_data_cache[symbol] = {
        'entry_price': entry_price,  # Giriş fiyatı
        'take_1': take_1,  # 1. kar alma seviyesi
        'take_2': take_2,  # 2. kar alma seviyesi
        'take_3': take_3,  # 3. kar alma seviyesi
        'take_4': take_4,  # 4. kar alma seviyesi
        'stop_loss': stop_loss,  # Zarar durdurma seviyesi
        'long': long,  # İşlem tipi (long/short)
        'investment': investment,  # Yatırım miktarı
        'expiry': datetime.now() + CACHE_EXPIRY_DURATION  # Cache geçerlilik süresi (örn. 60 dakika)
    }


def aggregate_signals(df):
    signals = {
        "rsi": rsi_analysis(df),
        "macd": macd_analysis(df),
        "adx": adx_analysis(df),
        "bollinger": bollinger_analysis(df),
        "stochastic_rsi": stochastic_rsi_analysis(df),
        "fibonacci": fibonacci_analysis(df),
        "obv": obv_analysis(df),
        "vwap": vwap_analysis(df)
    }

    buy_signals = sum(1 for sig in signals.values() if sig == "buy_signal")
    sell_signals = sum(1 for sig in signals.values() if sig == "sell_signal")

    if buy_signals >= 4:  # En az 3 buy_signal varsa
        return "long"
    elif sell_signals >= 4:  # En az 3 sell_signal varsa
        return "short"
    return "neutral"


# Mevcut fiyatı alma fonksiyonu
def get_current_price(symbol):
    url = f'https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}'
    response = requests.get(url)
    data = response.json()
    return float(data['price'])


# Cache kontrolü ve süresi dolmuş işlemleri temizleme
async def check_expired_cache():
    expired_coins = []
    for symbol in list(coin_data_cache.keys()):
        if datetime.now() > coin_data_cache[symbol]['expiry']:
            # await calculate_profit_or_loss(symbol)
            expired_coins.append(symbol)
    for symbol in expired_coins:
        del coin_data_cache[symbol]


async def calculate_profit_or_loss(symbol):
    if symbol in coin_data_cache:
        data = coin_data_cache[symbol]
        current_price = get_current_price(symbol)  # Mevcut fiyatı al
        entry_price = data['entry_price']
        long = data['long']
        investment = data['investment']
        leverage = 20  # Kaldıraç oranı

        # Kar/zarar oranını hesapla (long veya short pozisyon)
        if long:
            profit_loss_ratio = (current_price - entry_price) / entry_price
        else:
            profit_loss_ratio = (entry_price - current_price) / entry_price

        # Kaldıraçlı kar/zarar hesaplama
        profit_or_loss = investment * profit_loss_ratio * leverage
        result = "kar" if profit_or_loss > 0 else "zarar"

        # Telegram mesajı gönder
        message = f"{symbol} için işlem sona erdi.\n" \
                  f"Giriş fiyatı: ${entry_price:.8f}\n" \
                  f"Mevcut fiyat: ${current_price:.8f}\n" \
                  f"Sonuç: {result}.\n" \
                  f"Kaldıraçlı {result} edilen miktar: ${profit_or_loss:.2f} (20x Kaldıraç)"
        await send_telegram_message_loss(symbol, message)


# Telegram mesaj gönderme fonksiyonları
async def send_telegram_message(symbol, suggestion, support_proximity, resistance_proximity):
    support_message = f"Destek seviyesine %{support_proximity:.2f} yakın.\n" if support_proximity <= 0.5 else ""
    resistance_message = f"Direnç seviyesine %{resistance_proximity:.2f} yakın.\n" if resistance_proximity <= 0.5 else ""
    #await bot.send_message(chat_id=chat_id,
                           #text=f"🌹 {symbol} \n\n{support_message}{resistance_message}\n{suggestion}")
    await send_to_subscribers(f"🌹 {symbol} \n\n{support_message}{resistance_message}\n{suggestion}")


async def send_telegram_message_loss(symbol, suggestion):
    await bot.send_message(chat_id=chat_id, text=f"🌹 {symbol}  (30m/1h) \n\n{suggestion}")


# Stabil coin olup olmadığını kontrol eden fonksiyon
def is_stablecoin(symbol):
    normalized_symbol = symbol.replace('/', '')
    stable_symbols = [coin.replace('/', '') for coin in exclude_stablecoins]
    return normalized_symbol in stable_symbols


# Gözlem ve sinyal oluşturma fonksiyonu
async def generate_trade_suggestion(df, symbol):
    # Eğer bu sembol cache'de mevcutsa sinyal üretmeden çık
    if is_cached(symbol):
        print(f"{symbol} zaten cache'de, yeni sinyal üretilmeyecek.")
        return

    # Mevcut fiyatı al
    current_price = df['close'].iloc[-1]

    # Destek ve direnç seviyelerini bul
    support, resistance = find_support_resistance(df)

    # Destek ve direnç seviyelerine olan yakınlığı hesapla
    support_proximity = calculate_proximity(current_price, support)
    resistance_proximity = calculate_proximity(current_price, resistance)

    # İndikatörlerin sonuçlarına göre sinyal toplama
    signal = aggregate_signals(df)

    # Eğer sinyal "long" ise
    if signal == "long":
        entry_price = current_price
        take_1, take_2, take_3, take_4, stop_loss = calculate_take_profit_and_stop_loss(entry_price, long=True)
        suggestion = f"⏩ LONG \n" \
                     f"✳️ Giriş: {entry_price:.8f}\n" \
                     f"🥂 Hedefler: {take_1:.8f} - {take_2:.8f} - {take_3:.8f} - {take_4:.8f}\n" \
                     f"⚜️ Kaldıraç : Cross 20x\n" \
                     f"    destek: {support:.8f}\n" \
                     f"    Direnç: {resistance:.8f}\n" \
                     f"❌ Stop Loss: ${stop_loss:.8f}"
        # Sinyali Telegram'a gönder
        await send_telegram_message(symbol, suggestion, support_proximity, resistance_proximity)

        # Cache'e bu sembolü kaydet
        cache_coin_data(symbol, entry_price, take_1, take_2, take_3, take_4, stop_loss, long=True)

    # Eğer sinyal "short" ise
    elif signal == "short":
        entry_price = current_price
        take_1, take_2, take_3, take_4, stop_loss = calculate_take_profit_and_stop_loss(entry_price, long=False)
        suggestion = f"⏩ SHORT \n" \
                     f"✳️ Giriş: {entry_price:.8f}\n" \
                     f"🥂 Hedefler: {take_1:.8f} - {take_2:.8f} - {take_3:.8f} - {take_4:.8f}\n" \
                     f"⚜️ Kaldıraç : Cross 20x\n" \
                     f"    destek: {support:.8f}\n" \
                     f"    Direnç: {resistance:.8f}\n" \
                     f"❌ Stop Loss: ${stop_loss:.8f}"

        # Sinyali Telegram'a gönder
        await send_telegram_message(symbol, suggestion, support_proximity, resistance_proximity)

        # Cache'e bu sembolü kaydet
        cache_coin_data(symbol, entry_price, take_1, take_2, take_3, take_4, stop_loss, long=False)

    # Eğer sinyal yoksa
    else:
        print(f"{symbol}: Sinyal yok.")


'''
# Mevcut tüm coinleri izleyip sinyal üretme
async def monitor_all_coins():
    symbols = get_binance_futures_pairs()
    for symbol in symbols:
        try:
            await check_expired_cache()
            if is_stablecoin(symbol):
                continue
            df_dict = fetch_futures_data_multiple_intervals(symbol)
            for interval, df in df_dict.items():
                support, resistance = find_support_resistance(df)
                current_price = df['close'].iloc[-1]
                support_proximity = calculate_proximity(current_price, support)
                resistance_proximity = calculate_proximity(current_price, resistance)
                if resistance_proximity <= 0.5 or support_proximity <= 0.5:
                    await generate_trade_suggestion(df, symbol)
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Hata oluştu: {symbol}, {e}")
            await asyncio.sleep(30)
            continue
'''


# Mevcut tüm coinleri izleyip sinyal üretme
import traceback

async def monitor_all_coins():
    symbols = get_binance_futures_pairs()
    for symbol in symbols:
        try:
            await check_expired_cache()

            if is_stablecoin(symbol):
                continue

            df_dict = fetch_futures_data_multiple_intervals(symbol)

            for interval, df in df_dict.items():
                if df.empty or len(df) < 1:
                    print(f"{symbol} için yeterli veri yok: {interval} interval")
                    continue

                if 'close' in df.columns and len(df['close']) > 0:
                    current_price = df['close'].iloc[-1]
                else:
                    print(f"{symbol} için 'close' verisi mevcut değil.")
                    continue

                support, resistance = find_support_resistance(df)

                support_proximity = calculate_proximity(current_price, support)
                resistance_proximity = calculate_proximity(current_price, resistance)

                if resistance_proximity <= 0.5 or support_proximity <= 0.5:
                    await generate_trade_suggestion(df, symbol)

        except Exception as e:
            print(f"Hata oluştu: {symbol}, {e}")
            traceback.print_exc()  # 💥 Bu satır hatanın tam konumunu gösterir
            continue


# Botu başlatma
async def main():
    application = Application.builder().token(bot_token).build()
    application.add_handler(CommandHandler("start", start))
    
    # Start the bot and your monitoring loop
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    while True:
        await monitor_all_coins()
        await asyncio.sleep(30 * 15)


if __name__ == "__main__":
    asyncio.run(main())

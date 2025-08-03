import os
import time
import random
import logging
import pandas as pd
import numpy as np
import talib
from iqoptionapi.stable_api import IQ_Option
from dotenv import load_dotenv
import requests
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime

# Configurar logging
logging.basicConfig(
    filename="trade_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Carregar variáveis de ambiente
load_dotenv()
IQ_OPTION_EMAIL = os.getenv("IQ_OPTION_EMAIL")
IQ_OPTION_SENHA = os.getenv("IQ_OPTION_SENHA")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")
EXNOVA_API_KEY = os.getenv("EXNOVA_API_KEY")  # Suposta chave para Exnova

# Validação de credenciais
if not IQ_OPTION_EMAIL or not IQ_OPTION_SENHA:
    logging.error("Credenciais IQ Option não encontradas no arquivo .env")
    raise ValueError("Configure IQ_OPTION_EMAIL e IQ_OPTION_SENHA no arquivo .env")

# Parâmetros de negociação
PARES = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD"]
VALOR_INICIAL = 2.0
DIRECAO_PADRAO = "call"
ENTRADAS_MAX = 5
GALE = True
MAX_RETRIES = 5
MAX_BET = 100.0
TRADE_TIMEOUT = 70
MINIMUM_BALANCE = 1.0
RSI_PERIODO = 14
SMA_PERIODO = 20
EMA_PERIODO = 10
TAKE_PROFIT = 19  # Pips
STOP_LOSS = 10    # Pips
MODEL_PATH = "trading_model.pkl"
BACKTEST_CANDLES = 1000  # Quantidade de candles para backtesting

# Configuração de corretoras
BROKERS = {
    "iq_option": {"enabled": True, "api": None},
    "deriv": {"enabled": False, "api": None},
    "exnova": {"enabled": True, "api": None},
}

# Inicializar APIs
def initialize_brokers():
    try:
        if BROKERS["iq_option"]["enabled"]:
            BROKERS["iq_option"]["api"] = IQ_Option(IQ_OPTION_EMAIL, IQ_OPTION_SENHA)
            logging.info("API IQ Option inicializada")
        
        if BROKERS["deriv"]["enabled"] and DERIV_API_TOKEN:
            BROKERS["deriv"]["api"] = ccxt.deriv({"apiKey": DERIV_API_TOKEN})
            logging.info("API Deriv inicializada")
        
        if BROKERS["exnova"]["enabled"] and EXNOVA_API_KEY:
            # Exnova: Suposição de API similar à IQ Option ou WebSocket
            # Substitua pela implementação real da Exnova API
            BROKERS["exnova"]["api"] = ExnovaAPI(EXNOVA_API_KEY)  # Placeholder
            logging.info("API Exnova inicializada")
    except Exception as e:
        logging.error(f"Erro ao inicializar APIs: {e}")
        raise

# Placeholder para Exnova API (substituir pela implementação real)
class ExnovaAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        # Implementar conexão WebSocket ou API real aqui
        logging.warning("ExnovaAPI é um placeholder. Substitua pela API real.")
    
    def connect(self):
        # Implementar conexão real
        return True
    
    def check_connect(self):
        # Implementar verificação de conexão
        return True
    
    def get_balance(self):
        # Implementar obtenção de saldo
        return 1000.0  # Placeholder
    
    def buy(self, valor, par, direcao, expiracao):
        # Implementar negociação
        return True, "trade_id_placeholder"
    
    def check_win_v3(self, trade_id):
        # Implementar verificação de resultado
        return random.uniform(-10, 10)  # Placeholder
    
    def get_candles(self, par, timeframe, count, timestamp):
        # Implementar obtenção de candles
        return [{"close": 1.0, "open": 1.0, "high": 1.0, "low": 1.0, "volume": 100}]  # Placeholder
    
    def start_candles_stream(self, par, timeframe, count):
        pass
    
    def stop_candles_stream(self, par, timeframe):
        pass
    
    def change_balance(self, account_type):
        pass
    
    def disconnect(self):
        pass

# Função para conectar com retry
def connect_broker(broker_name, max_retries):
    api = BROKERS[broker_name]["api"]
    for attempt in range(1, max_retries + 1):
        try:
            if broker_name in ["iq_option", "exnova"]:
                api.connect()
                if api.check_connect():
                    logging.info(f"Conexão estabelecida com {broker_name}")
                    return True
            elif broker_name == "deriv":
                api.fetch_balance()
                logging.info(f"Conexão estabelecida com {broker_name}")
                return True
            time.sleep(2)
        except Exception as e:
            logging.error(f"Erro na conexão com {broker_name} (tentativa {attempt}/{max_retries}): {e}")
            time.sleep(2)
    logging.error(f"Falha ao conectar com {broker_name} após {max_retries} tentativas")
    return False

# Obter dados de mercado
def get_market_data(broker_name, par, timeframe="1m", candles=200):
    try:
        api = BROKERS[broker_name]["api"]
        if broker_name in ["iq_option", "exnova"]:
            api.start_candles_stream(par, 60, candles)
            candles_data = api.get_candles(par, 60, candles, time.time())
            api.stop_candles_stream(par, 60)
            df = pd.DataFrame(candles_data)
            df["close"] = df["close"].astype(float)
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["volume"] = df["volume"].astype(float)
            return df
        elif broker_name == "deriv":
            ohlcv = api.fetch_ohlcv(par, timeframe, limit=candles)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            return df
    except Exception as e:
        logging.error(f"Erro ao obter dados de mercado para {par} em {broker_name}: {e}")
        return None

# Preparar dados para o modelo
def prepare_features(df):
    try:
        df["rsi"] = talib.RSI(df["close"], timeperiod=RSI_PERIODO)
        df["sma"] = talib.SMA(df["close"], timeperiod=SMA_PERIODO)
        df["ema"] = talib.EMA(df["close"], timeperiod=EMA_PERIODO)
        df["volatility"] = df["close"].rolling(window=20).std()
        df["price_change"] = df["close"].pct_change()
        df["candle_size"] = (df["close"] - df["open"]) / df["open"]
        
        # Criar target: 1 (call), -1 (put), 0 (neutro)
        df["target"] = 0
        df.loc[df["close"].shift(-1) > df["close"] + TAKE_PROFIT * 0.0001, "target"] = 1
        df.loc[df["close"].shift(-1) < df["close"] - STOP_LOSS * 0.0001, "target"] = -1
        
        df = df.dropna()
        return df
    except Exception as e:
        logging.error(f"Erro ao preparar features: {e}")
        return None

# Treinar ou carregar modelo
def train_or_load_model(df, retrain=False):
    model_file = MODEL_PATH
    if os.path.exists(model_file) and not retrain:
        model = joblib.load(model_file)
        logging.info("Modelo carregado de %s", model_file)
        return model
    
    try:
        features = ["rsi", "sma", "ema", "volatility", "price_change", "candle_size"]
        X = df[features]
        y = df["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Acurácia do modelo: {accuracy:.2f}")
        
        joblib.dump(model, model_file)
        logging.info("Modelo salvo em %s", model_file)
        return model
    except Exception as e:
        logging.error(f"Erro ao treinar modelo: {e}")
        return None

# Prever sinal com o modelo
def predict_signal(model, df):
    try:
        features = ["rsi", "sma", "ema", "volatility", "price_change", "candle_size"]
        X = df[features].iloc[-1:].values
        prediction = model.predict(X)[0]
        if prediction == 1:
            return "call"
        elif prediction == -1:
            return "put"
        return None
    except Exception as e:
        logging.error(f"Erro ao prever sinal: {e}")
        return None

# Função de backtesting
def backtest_model(model, df, par, initial_balance=1000.0):
    try:
        balance = initial_balance
        trades = []
        wins = 0
        losses = 0
        max_drawdown = 0
        peak_balance = balance
        
        features = ["rsi", "sma", "ema", "volatility", "price_change", "candle_size"]
        
        for i in range(len(df) - 1):
            X = df[features].iloc[i:i+1].values
            if np.any(np.isnan(X)):
                continue
            
            prediction = model.predict(X)[0]
            if prediction == 0:
                continue
            
            direcao = "call" if prediction == 1 else "put"
            valor = VALOR_INICIAL
            
            # Simular resultado da negociação
            actual_price = df["close"].iloc[i]
            next_price = df["close"].iloc[i + 1]
            profit = 0
            
            if direcao == "call" and next_price > actual_price + TAKE_PROFIT * 0.0001:
                profit = valor * 0.85  # Supondo payout de 85%
                wins += 1
            elif direcao == "put" and next_price < actual_price - STOP_LOSS * 0.0001:
                profit = valor * 0.85
                wins += 1
            else:
                profit = -valor
                losses += 1
            
            balance += profit
            trades.append({"par": par, "direcao": direcao, "valor": valor, "profit": profit, "balance": balance})
            
            # Calcular drawdown
            peak_balance = max(peak_balance, balance)
            drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, drawdown)
            
            # Aplicar Martingale no backtest
            if profit < 0 and GALE:
                valor = min(valor * 2, MAX_BET)
            else:
                valor = VALOR_INICIAL
        
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        final_balance = balance
        total_trades = wins + losses
        
        logging.info(f"Backtest para {par}:")
        logging.info(f"Saldo Inicial: {initial_balance:.2f}")
        logging.info(f"Saldo Final: {final_balance:.2f}")
        logging.info(f"Total de Trades: {total_trades}")
        logging.info(f"Taxa de Acerto: {win_rate:.2%}")
        logging.info(f"Max Drawdown: {max_drawdown:.2%}")
        
        return pd.DataFrame(trades), final_balance, win_rate, max_drawdown
    except Exception as e:
        logging.error(f"Erro no backtest para {par}: {e}")
        return None, initial_balance, 0, 0

# Função para realizar negociação
def entrar(broker_name, par, valor, direcao):
    try:
        api = BROKERS[broker_name]["api"]
        if broker_name in ["iq_option", "exnova"]:
            balance = api.get_balance()
            if balance < MINIMUM_BALANCE or valor > balance:
                logging.error(f"Saldo insuficiente em {broker_name}: {balance} < {valor}")
                return None
            status, trade_id = api.buy(valor, par, direcao, 1)
            if not status:
                logging.error(f"Falha ao realizar negociação em {broker_name}: {trade_id}")
                return None
            logging.info(f"Negociação realizada em {broker_name}: {direcao} com valor {valor} em {par}")
            
            start_time = time.time()
            while time.time() - start_time < TRADE_TIMEOUT:
                result = api.check_win_v3(trade_id)
                if result is not None:
                    profit = float(result)
                    logging.info(f"Resultado da negociação em {broker_name}: {profit:.2f}")
                    return profit
                time.sleep(1)
            logging.warning(f"Timeout ao aguardar resultado em {broker_name} para {par}")
            return None
        elif broker_name == "deriv":
            order = api.create_order(symbol=par, type="market", side=direcao, amount=valor)
            logging.info(f"Negociação realizada em {broker_name}: {direcao} com valor {valor} em {par}")
            return order["id"]
    except Exception as e:
        logging.error(f"Erro durante negociação em {broker_name} para {par}: {e}")
        return None

# Loop principal
def main():
    initialize_brokers()
    
    # Conectar a todas as corretoras habilitadas
    for broker_name in BROKERS:
        if BROKERS[broker_name]["enabled"]:
            if not connect_broker(broker_name, MAX_RETRIES):
                logging.error(f"Não foi possível conectar à {broker_name}, desativando")
                BROKERS[broker_name]["enabled"] = False
    
    try:
        for broker_name in BROKERS:
            if not BROKERS[broker_name]["enabled"]:
                continue
            if broker_name in ["iq_option", "exnova"]:
                BROKERS[broker_name]["api"].change_balance("PRACTICE")
                logging.info(f"Saldo alterado para conta de prática em {broker_name}")

        # Coletar dados para treinamento e backtesting
        df_training = None
        for par in PARES:
            df = get_market_data("iq_option", par, candles=BACKTEST_CANDLES)
            if df is not None:
                df = prepare_features(df)
                if df is not None:
                    df_training = df if df_training is None else pd.concat([df_training, df])
        
        if df_training is None or df_training.empty:
            logging.error("Não foi possível coletar dados para treinamento")
            return
        
        # Treinar ou carregar modelo
        model = train_or_load_model(df_training, retrain=False)
        if model is None:
            logging.error("Falha ao carregar ou treinar modelo, encerrando")
            return

        # Executar backtesting para cada par
        for par in PARES:
            df = get_market_data("iq_option", par, candles=BACKTEST_CANDLES)
            if df is not None:
                df = prepare_features(df)
                if df is not None:
                    trades_df, final_balance, win_rate, max_drawdown = backtest_model(model, df, par)
                    if trades_df is not None:
                        trades_df.to_csv(f"backtest_{par}.csv")
                        logging.info(f"Backtest salvo em backtest_{par}.csv")

        # Loop de negociação ao vivo
        for i in range(ENTRADAS_MAX):
            for par in PARES:
                logging.info(f"Iniciando análise para {par} (entrada {i + 1}/{ENTRADAS_MAX})")
                
                for broker_name in BROKERS:
                    if not BROKERS[broker_name]["enabled"]:
                        continue
                    df = get_market_data(broker_name, par)
                    if df is None or df.empty:
                        logging.warning(f"Sem dados de mercado para {par} em {broker_name}")
                        continue

                    df = prepare_features(df)
                    if df is None or df.empty:
                        logging.warning(f"Sem features válidas para {par} em {broker_name}")
                        continue

                    direcao = predict_signal(model, df)
                    if not direcao:
                        logging.info(f"Sem sinal de negociação para {par} em {broker_name}")
                        continue
                    
                    valor = VALOR_INICIAL
                    result = entrar(broker_name, par, valor, direcao)
                    
                    if result is None:
                        logging.error(f"Negociação falhou para {par} em {broker_name}")
                        continue
                    
                    if result < 0 and GALE:
                        new_valor = min(valor * 2, MAX_BET)
                        logging.info(f"Perda detectada, nova aposta: {new_valor:.2f} em {par}")
                        valor = new_valor
                    else:
                        valor = VALOR_INICIAL
                        logging.info(f"Resetando aposta para: {valor:.2f} em {par}")
                    
                    time.sleep(random.uniform(2, 5))
    
    except Exception as e:
        logging.error(f"Erro no loop principal: {e}")
    finally:
        for broker_name in BROKERS:
            if BROKERS[broker_name]["enabled"] and BROKERS[broker_name]["api"]:
                try:
                    if broker_name in ["iq_option", "exnova"]:
                        BROKERS[broker_name]["api"].disconnect()
                    logging.info(f"Conexão com {broker_name} encerrada")
                except Exception as e:
                    logging.error(f"Erro ao desconectar {broker_name}: {e}")
        logging.info("Script finalizado")

if __name__ == "__main__":
    main()

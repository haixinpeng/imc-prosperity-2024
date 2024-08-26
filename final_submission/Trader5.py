import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import jsonpickle
import numpy as np
import math
from statistics import NormalDist

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."


logger = Logger()


class Trader:
    # Define class variables that won't be updated
    POS_LIMIT = {'STARFRUIT': 20, 'AMETHYSTS': 20, "ORCHIDS": 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60,
                 'GIFT_BASKET': 60, 'COCONUT_COUPON': 600, 'COCONUT': 300}
    sf_cache_d = 4
    b_ma = 365
    b_std = 79.5
    c_std = 13.529247
    opn_daily_vol = 0.010119297003746032

    # Calculate VWAP mid
    def calc_vwap(self, order_depth):
        vwap_bid = 0
        total_bid = 0
        vwap_ask = 0
        total_ask = 0

        for key, value in order_depth.buy_orders.items():
            if key and value and isinstance(key, int) and isinstance(value, int):
                vwap_bid += int(key) * int(value)
                total_bid += int(value)

        for key, value in order_depth.sell_orders.items():
            if key and value and isinstance(key, int) and isinstance(value, int):
                vwap_ask += int(key) * int(value)
                total_ask += int(value)

        if total_bid != 0 and total_ask != 0:
            vwap_mid = (vwap_bid / total_bid + vwap_ask / total_ask) / 2
        else:
            vwap_mid = 0

        return round(vwap_mid)

    # Run regression for STARFRUIT
    def run_regression(self, sf_cache, sf_error):
        coef = [-0.33105, -0.09977, -0.99998]
        past_vals = sf_cache[::-1]
        t0 = (past_vals[0] - 2 * past_vals[1] + past_vals[2]) * coef[0]
        t1 = (past_vals[1] - 2 * past_vals[2] + past_vals[3]) * coef[1]
        t2 = coef[2] * sf_error
        t3 = 2 * past_vals[0] - past_vals[1]

        return int(round(t0 + t1 + t2 + t3))

    def find_error(self, realized_val, sf_cache, sf_error):
        coef = [-0.33105, -0.09977, -0.99998]
        past_vals = sf_cache[::-1]
        t0 = (past_vals[0] - 2 * past_vals[1] + past_vals[2]) * coef[0]
        t1 = (past_vals[1] - 2 * past_vals[2] + past_vals[3]) * coef[1]
        t2 = coef[2] * sf_error
        t3 = 2 * past_vals[0] - past_vals[1]
        error = (realized_val - t3 - (t0 + t1 + t2))

        return error

    # Find best or worst sell and buy price using type = 1 or -1
    def find_best(self, order_depth, type=1):
        if type == 1:
            ask, _ = sorted(order_depth.sell_orders.items())[0]
            bid, _ = sorted(order_depth.buy_orders.items(), reverse=True)[0]
        else:
            ask, _ = sorted(order_depth.sell_orders.items())[-1]
            bid, _ = sorted(order_depth.buy_orders.items(), reverse=True)[-1]

        return int(round(bid)), int(round(ask))

    def get_vol(self, order_depth):
        buy_vol = 0
        sell_vol = 0
        for price, vol in order_depth.buy_orders.items():
            buy_vol += vol
        for price, vol in order_depth.sell_orders.items():
            sell_vol += vol

        return abs(buy_vol), abs(sell_vol)

    def get_traders_orders(self, state: TradingState, trader_name: str, product: str):
        trade = 0
        market_trades = state.market_trades.get(product, [])
        if market_trades:
            for t in market_trades:
                if t.buyer == trader_name:
                    trade += t.quantity
                elif t.seller == trader_name:
                    trade -= t.quantity
        return trade

    def get_own_orders(self, state: TradingState, trader_name: str, product: str):
        trade = 0
        own_trades = state.own_trades.get(product, [])
        if own_trades:
            for t in own_trades:
                if t.buyer == trader_name:
                    trade += t.quantity
                elif t.seller == trader_name:
                    trade -= t.quantity
        return trade

    # STARFRUIT Strategy
    def compute_sf_order(self, position, LIMIT, order_depth, sf_cache, sf_next_price, sf_error):
        orders: list[Order] = []

        # Define current position and product
        product = 'STARFRUIT'
        cpos = position
        logger.print("New iteration! Current pos is {}".format(cpos))

        # Calculate current vwap mid price
        sf_vwap_mid = self.calc_vwap(order_depth)
        if sf_vwap_mid == 0:
            sf_vwap_mid = sf_cache[-1]

        # Update error term before updating cache
        if sf_next_price is not None:
            sf_error = self.find_error(sf_vwap_mid, sf_cache, sf_error)

        # Update cache
        if len(sf_cache) == self.sf_cache_d:
            sf_cache.pop(0)
        sf_cache.append(sf_vwap_mid)

        INF = 1e9
        sf_lb = -INF
        sf_ub = INF

        # Only predict when we have enough observations
        if len(sf_cache) == self.sf_cache_d:
            sf_next_price = self.run_regression(sf_cache, sf_error)
            sf_lb = sf_next_price - 1
            sf_ub = sf_next_price + 1

        # Find best bid/ask
        best_bid, best_ask = self.find_best(order_depth)
        undercut_bid = best_bid + 1
        undercut_ask = best_ask - 1
        bid_pr = min(undercut_bid, sf_lb)  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_ask, sf_ub)

        # Market-taking buy orders
        for price, vol in sorted(order_depth.sell_orders.items()):
            if (price <= sf_lb) or (position < 0 and price == sf_lb + 1) and (cpos < LIMIT):
                qty = min(-vol, LIMIT - cpos)
                cpos += qty
                orders.append(Order(product, price, qty))
                logger.print(f"Sent market buy order {product},{price},{qty}")

        # Market-making buy orders
        if len(sf_cache) == self.sf_cache_d and cpos < LIMIT:
            qty = LIMIT - cpos
            orders.append(Order(product, bid_pr, qty))

        # Re-initialze current position as position limit only applies to aggregated buy/sell orders
        cpos = position

        # Market-taking sell orders
        for price, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if (price >= sf_ub) or (position > 0 and price == sf_ub - 1) and (cpos > -LIMIT):
                qty = max(-vol, -LIMIT - cpos)
                cpos += qty
                orders.append(Order(product, price, qty))
                logger.print(f"Sent market sell order {product},{price},{qty}")

        # Market-making sell orders
        if len(sf_cache) == self.sf_cache_d and cpos > -LIMIT:
            qty = -LIMIT - cpos
            orders.append(Order(product, sell_pr, qty))

        return orders, sf_cache, sf_next_price, sf_error

    # AMETHYSTS Strategy
    def compute_a_order(self, position, LIMIT, order_depth):
        orders: list[Order] = []

        # Define fair value of AMETHYSTS
        fair_value = 10000

        # Define current position and product
        product = 'AMETHYSTS'
        cpos = position

        # Find best bid/ask
        best_bid, best_ask = self.find_best(order_depth)
        undercut_bid = best_bid + 1
        undercut_ask = best_ask - 1

        # Market-taking buy orders
        for price, vol in sorted(order_depth.sell_orders.items()):
            if (price < fair_value) or (position < 0 and price == fair_value) and (cpos < LIMIT):
                qty = min(-vol, LIMIT - cpos)
                cpos += qty
                orders.append(Order(product, price, qty))

        # Place undercut orders
        if cpos < LIMIT:
            qty = min(40, LIMIT - cpos)
            if position < 0:
                orders.append(Order(product, min(undercut_bid + 1, fair_value - 1), qty))
            elif position > 15:
                orders.append(Order(product, min(undercut_bid - 1, fair_value - 1), qty))
            else:
                orders.append(Order(product, min(undercut_bid, fair_value - 1), qty))

        # Re-initialize current position as position limit only applies to aggregated buy/sell orders
        cpos = position

        # Market-taking sell orders
        for price, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if (price > fair_value) or (position > 0 and price == fair_value) and (cpos > -LIMIT):
                qty = max(-vol, -LIMIT - cpos)
                cpos += qty
                orders.append(Order(product, price, qty))

        # Market-making sell orders
        if cpos > -LIMIT:
            qty = max(-40, -LIMIT - cpos)
            if position > 0:
                orders.append(Order(product, max(undercut_ask - 1, fair_value + 1), qty))
            elif position < -15:
                orders.append(Order(product, max(undercut_ask + 1, fair_value + 1), qty))
            else:
                orders.append(Order(product, max(undercut_ask, fair_value + 1), qty))

        return orders

    # ORCHIDS Strategy
    def compute_o_order(self, position, LIMIT, order_depth, obs):
        orders: list[Order] = []

        # Define current position and product and observations
        product = 'ORCHIDS'
        cpos = position
        conversion_obs = obs.conversionObservations
        obs = conversion_obs['ORCHIDS']

        # Define threshold to play market arb order
        market_o_threshold = 2

        # Find best bid and ask
        best_bid, best_ask = self.find_best(order_depth)

        # Sell to the south
        for price, vol in sorted(order_depth.sell_orders.items()):
            if (cpos < LIMIT) and price + obs.transportFees + obs.exportTariff < obs.bidPrice:
                qty = min(-vol, LIMIT - cpos)
                cpos += qty
                orders.append(Order(product, price, qty))

        # Market making buy, and then sell to the south
        if (cpos < LIMIT):
            qty = LIMIT - cpos
            if obs.bidPrice - obs.transportFees - obs.exportTariff - (best_bid + 1) > 0:
                diff = obs.bidPrice - obs.transportFees - obs.exportTariff - best_bid
                orders.append(Order(product, int(best_bid + diff * 3 / 4), qty))

        # Buy from south
        cpos = 0
        for price, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if (cpos > -LIMIT) and price - (obs.askPrice + obs.transportFees + obs.importTariff) > market_o_threshold:
                qty = max(-vol, (-LIMIT - cpos))
                cpos += qty
                orders.append(Order(product, price, qty))

        # Market making sell, and then buy back from south
        if (cpos > -LIMIT):
            qty = -LIMIT - cpos
            if (best_ask - 1) - (obs.transportFees + obs.importTariff + obs.askPrice) > 0:
                diff = best_ask - (obs.transportFees + obs.importTariff + obs.askPrice)
                orders.append(Order(product, int(best_ask - diff * 3 / 4), qty))

        # logger.print("orders: ", orders)
        return orders

    # Basket Strategy
    def compute_b_order(self, position, order_depths, b_dir_bet):
        # CAVEATS: what if there is sometime the limit order is empty
        orders = {'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': []}
        prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']

        pos, osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell, vwap_mid = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = sorted(order_depths[p].sell_orders.items())
            obuy[p] = sorted(order_depths[p].buy_orders.items(), reverse=True)
            pos[p] = position.get(p,0)
            best_buy[p], best_sell[p] = self.find_best(order_depths[p])
            worst_buy[p], worst_sell[p] = self.find_best(order_depths[p], -1)
            vwap_mid[p] = self.calc_vwap(order_depths[p])

        prm = vwap_mid['GIFT_BASKET'] - vwap_mid['STRAWBERRIES'] * 6 - vwap_mid['CHOCOLATE'] * 4 - vwap_mid['ROSES']

        upper = self.b_ma + 0.5*self.b_std
        lower = self.b_ma - 0.5*self.b_std


        if prm >= upper:
            # short signal
            if (b_dir_bet == -1) and (position.get("STRAWBERRIES", 0) == self.POS_LIMIT["STRAWBERRIES"]):
                pass
            else:
                self.b_dir_bet = -1

                # Calculate available market volume to trade
                _,gb_best_buy_vol = obuy['GIFT_BASKET'][0]
                _, sb_sell_vol = osell['STRAWBERRIES'][0]
                _, c_sell_vol = osell['CHOCOLATE'][0]
                _, r_sell_vol = osell['ROSES'][0]
                sb_sell_vol = abs(sb_sell_vol)
                c_sell_vol = abs(c_sell_vol)
                r_sell_vol = abs(r_sell_vol)

                max_gb_qty = max(self.POS_LIMIT['GIFT_BASKET'] + pos['GIFT_BASKET'], 0)
                max_r_qty = max(self.POS_LIMIT['ROSES'] - pos['ROSES'],0 )
                max_c_qty = math.floor(max((self.POS_LIMIT['CHOCOLATE'] - pos['CHOCOLATE']),0) / 4)
                max_sb_qty = math.floor(max(self.POS_LIMIT['STRAWBERRIES'] - pos['STRAWBERRIES'],0)/ 6)

                # Compute the maximum permissible quantity respecting market volume and position limits
                basket_qty = min(gb_best_buy_vol, max_gb_qty, r_sell_vol, max_r_qty, math.floor(c_sell_vol / 4), max_c_qty,
                                 math.floor(sb_sell_vol / 6), max_sb_qty, int(prm-upper+1))

                if basket_qty > 0:
                    # logger.print("Here!! Short")
                    # Place order depending on current position
                    qty = basket_qty
                    orders['GIFT_BASKET'].append(Order('GIFT_BASKET', best_buy['GIFT_BASKET'], -qty))
                    orders['ROSES'].append(Order('ROSES', best_sell['ROSES'], qty))
                    orders['CHOCOLATE'].append(Order('CHOCOLATE', best_sell['CHOCOLATE'], 4 * qty))
                    orders['STRAWBERRIES'].append(Order('STRAWBERRIES', best_sell['STRAWBERRIES'], 6 * qty))

        elif prm <= lower:
            # long signal
            if (b_dir_bet == 1) and (position.get("STRAWBERRIES", 0) == -self.POS_LIMIT["STRAWBERRIES"]):
                pass
            else:
                b_dir_bet = 1

                # Calculate available market volume to trade
                _, gb_best_sell_vol = osell['GIFT_BASKET'][0]
                gb_best_sell_vol = abs(gb_best_sell_vol)
                sb_buy_vol, _ = obuy['STRAWBERRIES'][0]
                c_buy_vol, _ = obuy['CHOCOLATE'][0]
                r_buy_vol, _ = obuy['ROSES'][0]

                max_gb_qty = max(self.POS_LIMIT['GIFT_BASKET'] - pos['GIFT_BASKET'], 0)
                max_r_qty = max(self.POS_LIMIT['ROSES'] + pos['ROSES'],0 )
                max_c_qty = math.floor(max((self.POS_LIMIT['CHOCOLATE'] + pos['CHOCOLATE']),0) / 4)
                max_sb_qty = math.floor(max(self.POS_LIMIT['STRAWBERRIES'] + pos['STRAWBERRIES'],0)/ 6)

                # Compute the maximum permissible quantity respecting market volume and position limits
                basket_qty = min(gb_best_sell_vol, max_gb_qty, r_buy_vol, max_r_qty, math.floor(c_buy_vol / 4), max_c_qty,
                                 math.floor(sb_buy_vol / 6), max_sb_qty, int(lower-prm+1))

                if basket_qty > 0:
                    # print("Here!! long")
                    # Place order depending on current position
                    qty = basket_qty
                    orders['GIFT_BASKET'].append(Order('GIFT_BASKET', best_sell['GIFT_BASKET'], qty))
                    orders['ROSES'].append(Order('ROSES', best_buy['ROSES'], -qty))
                    orders['CHOCOLATE'].append(Order('CHOCOLATE', best_buy['CHOCOLATE'], 4 * -qty))
                    orders['STRAWBERRIES'].append(Order('STRAWBERRIES', best_buy['STRAWBERRIES'], 6 * -qty))


        return orders, b_dir_bet

    # Option Strategy
    def compute_opn_order(self, timestamp, position_opn, position_cc, order_depth_opn, order_depth_cc):
        orders = {'COCONUT_COUPON': [], 'COCONUT': []}

        # For simplicity, norm.cdf((cc['S/K'] - daily_vol**2 * (250-opn['day']) / 2) / (daily_vol * np.sqrt(250-opn['day'])))
        best_bid_cc, best_ask_cc = self.find_best(order_depth_cc)
        mid_cc = (best_bid_cc + best_ask_cc) / 2
        _,opn_best_sell_vol = sorted(order_depth_opn.sell_orders.items())[0]
        _, opn_best_buy_vol = sorted(order_depth_opn.buy_orders.items(), reverse=True)[0]

        best_bid_opn, best_ask_opn = self.find_best(order_depth_opn)
        worst_bid_opn, worst_ask_opn = self.find_best(order_depth_opn, -1)
        mid_opn = (best_bid_opn + best_ask_opn) / 2

        new_d1 = (np.log(mid_cc / 10000) + self.opn_daily_vol ** 2 * 245 / 2) / (self.opn_daily_vol * np.sqrt(245))
        new_d2 = (np.log(mid_cc / 10000) - self.opn_daily_vol ** 2 * 245 / 2) / (self.opn_daily_vol * np.sqrt(245))
        fair_value = NormalDist(mu=0, sigma=1).cdf(new_d1) * mid_cc - 10000 * NormalDist(mu=0, sigma=1).cdf(new_d2)

        # Calculate spread signal
        spread = (mid_opn - fair_value)

        # Open position or close position
        new_delta = NormalDist(mu=0, sigma=1).cdf(new_d1)
        qty = 0
        if spread > 0.5*self.c_std:
            if spread > 0.7*self.c_std:
                qty = -self.POS_LIMIT['COCONUT_COUPON'] - position_opn
            else:
                qty = max(-self.POS_LIMIT['COCONUT_COUPON'] - position_opn,-14)
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_bid_opn, qty))
        elif spread < -0.5*self.c_std:
            if spread < -0.7*self.c_std:
                qty = self.POS_LIMIT['COCONUT_COUPON'] - position_opn
            else:
                qty = min(self.POS_LIMIT['COCONUT_COUPON'] - position_opn,14)
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_ask_opn, qty))
        elif spread < -0.375*self.c_std and position_opn < 0:
            qty = - position_opn
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_ask_opn, qty))
        elif spread > 0.375*self.c_std and position_opn > 0:
            qty = - position_opn
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_bid_opn, qty))

        # Delta hedge
        if qty != 0:
            extra_delta = int(new_delta * (position_opn + qty)) - position_cc
            if extra_delta > 0:
                cc_qty = max(-extra_delta, -self.POS_LIMIT['COCONUT'] - position_cc)
                orders['COCONUT'].append(Order('COCONUT', best_bid_cc, cc_qty))
            elif extra_delta < 0:
                cc_qty = min(-extra_delta, self.POS_LIMIT['COCONUT'] - position_cc)
                orders['COCONUT'].append(Order('COCONUT', best_ask_cc, cc_qty))

        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:

        orders = {'GIFT_BASKET': [], 'ROSES': [], 'CHOCOLATE': [], 'STRAWBERRIES': []}
        conversions = 0

        # Initialize variables if timestamp == 0, else read from pickeled data
        if state.timestamp == 0:
            sf_cache = []
            sf_error = 0  # initial error term (best guess)
            sf_next_price = None
            b_dir_bet = 0
            trader_position = {'Rhianna': {'GIFT_BASKET':0, 'COCONUT':0},
                               'Raj': {'COCONUT':0},
                               'Vladimir': {'COCONUT':0},
                               'Valentina': {'STARFRUIT': 0}}
        else:
            data = jsonpickle.decode(state.traderData)
            sf_cache = data['sf_cache']
            sf_error = data['sf_error']
            sf_next_price = data['sf_next_price']
            b_dir_bet = data['b_dir_bet']
            trader_position = data['trader_position']

        for product in state.order_depths.keys():
            # Calculate orders for STARFRUIT
            if product == 'STARFRUIT':
                if len(state.order_depths[product].sell_orders) != 0 and len(
                        state.order_depths[product].buy_orders) != 0:
                    o, sf_cache, sf_next_price, sf_error = self.compute_sf_order(state.position.get('STARFRUIT', 0),
                                                                                 self.POS_LIMIT['STARFRUIT'],
                                                                                 state.order_depths['STARFRUIT'],
                                                                                 sf_cache, sf_next_price, sf_error)
                else:
                    o = []
                orders['STARFRUIT'] = o

            # Calculate orders for AMETHYSTS
            if product == 'AMETHYSTS':
                if len(state.order_depths[product].sell_orders) != 0 and len(
                        state.order_depths[product].buy_orders) != 0:
                    o = self.compute_a_order(state.position.get('AMETHYSTS', 0), self.POS_LIMIT['AMETHYSTS'],
                                             state.order_depths['AMETHYSTS'])

                else:
                    o = []
                orders['AMETHYSTS'] = o

            # Calculate orders for ORCHIDS
            if product == 'ORCHIDS':
                if len(state.order_depths[product].sell_orders) != 0 and len(
                        state.order_depths[product].buy_orders) != 0:
                    o = self.compute_o_order(state.position.get('ORCHIDS', 0), self.POS_LIMIT['ORCHIDS'],
                                             state.order_depths['ORCHIDS'], state.observations)
                else:
                    o = []
                orders['ORCHIDS'] = o

            if product == 'COCONUT_COUPON':
                if len(state.order_depths[product].sell_orders) != 0 and len(
                        state.order_depths[product].buy_orders) != 0:
                    o = self.compute_opn_order(state.timestamp, state.position.get(product, 0), state.position.get('COCONUT', 0),
                                            state.order_depths[product], state.order_depths['COCONUT'])
                else:
                    o = []
                orders['COCONUT_COUPON'] = o['COCONUT_COUPON']
                orders['COCONUT'] = o['COCONUT']

        # Calculate orders for basket strategy
        basket_o, b_dir_bet = self.compute_b_order(state.position, state.order_depths,b_dir_bet)
        orders['GIFT_BASKET'] = basket_o['GIFT_BASKET']
        orders['ROSES'] = basket_o['ROSES']
        orders['CHOCOLATE'] = basket_o['CHOCOLATE']
        orders['STRAWBERRIES'] = basket_o['STRAWBERRIES']


        # Update trader positions
        for trader in trader_position.keys():
            for product in trader_position[trader]:
                qty = self.get_traders_orders(state, trader, product)
                trader_position[trader][product] += qty
                qty = self.get_own_orders(state, trader, product)
                trader_position[trader][product] += qty

        logger.print("Positions: ", trader_position)
        logger.print("Own: ", state.own_trades)

        # Combine data into a single object, e.g., a dictionary and encode data into string
        combined_data = {'sf_cache': sf_cache, 'sf_next_price': sf_next_price, 'sf_error': sf_error, 'b_dir_bet': b_dir_bet, 'trader_position': trader_position}

        # Serialize the combined object to a JSON-formatted string
        trader_data = jsonpickle.encode(combined_data)

        logger.flush(state, orders, conversions, trader_data)
        conversions = -state.position.get('ORCHIDS',0)  # in the arbitrage strategy, we wanna offset our position immediately

        return orders, conversions, trader_data

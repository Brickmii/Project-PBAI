import tkinter as tk
from tkinter import messagebox
import random


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def value(self):
        if self.rank in ['J', 'Q', 'K']:
            return 10
        elif self.rank == 'A':
            return 11
        else:
            return int(self.rank)


class Deck:
    suits = ['♠', '♥', '♦', '♣']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    def __init__(self, num_decks=6):
        self.cards = []
        for _ in range(num_decks):
            for suit in self.suits:
                for rank in self.ranks:
                    self.cards.append(Card(suit, rank))
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop() if self.cards else None


class Hand:
    def __init__(self):
        self.cards = []
        self.bet = 0
        self.is_split = False
        self.is_doubled = False
        self.is_finished = False

    def add_card(self, card):
        self.cards.append(card)

    def get_value(self):
        value = sum(card.value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == 'A')

        while value > 21 and aces:
            value -= 10
            aces -= 1

        return value

    def is_blackjack(self):
        return len(self.cards) == 2 and self.get_value() == 21

    def is_busted(self):
        return self.get_value() > 21

    def can_split(self):
        return len(self.cards) == 2 and self.cards[0].value() == self.cards[1].value()

    def can_double(self):
        return len(self.cards) == 2


class BlackjackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Casino Blackjack")
        self.root.geometry("900x700")
        self.root.configure(bg='#0B6623')

        self.deck = Deck(6)
        self.player_hands = []
        self.current_hand_index = 0
        self.dealer_hand = Hand()
        self.balance = 1000
        self.current_bet = 0
        self.insurance_bet = 0
        self.game_active = False

        self.setup_ui()

    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="BLACKJACK", font=('Arial', 28, 'bold'),
                         bg='#0B6623', fg='gold')
        title.pack(pady=10)

        # Balance display
        self.balance_label = tk.Label(self.root, text=f"Balance: ${self.balance}",
                                      font=('Arial', 18, 'bold'), bg='#0B6623', fg='white')
        self.balance_label.pack()

        # Dealer section
        dealer_frame = tk.Frame(self.root, bg='#0B6623')
        dealer_frame.pack(pady=20)

        tk.Label(dealer_frame, text="DEALER", font=('Arial', 16, 'bold'),
                 bg='#0B6623', fg='white').pack()

        self.dealer_cards_label = tk.Label(dealer_frame, text="", font=('Arial', 24),
                                           bg='#0B6623', fg='white')
        self.dealer_cards_label.pack()

        self.dealer_value_label = tk.Label(dealer_frame, text="", font=('Arial', 14),
                                           bg='#0B6623', fg='yellow')
        self.dealer_value_label.pack()

        # Player section
        player_frame = tk.Frame(self.root, bg='#0B6623')
        player_frame.pack(pady=20)

        tk.Label(player_frame, text="PLAYER", font=('Arial', 16, 'bold'),
                 bg='#0B6623', fg='white').pack()

        self.player_cards_label = tk.Label(player_frame, text="", font=('Arial', 24),
                                           bg='#0B6623', fg='white')
        self.player_cards_label.pack()

        self.player_value_label = tk.Label(player_frame, text="", font=('Arial', 14),
                                           bg='#0B6623', fg='yellow')
        self.player_value_label.pack()

        # Betting section
        bet_frame = tk.Frame(self.root, bg='#0B6623')
        bet_frame.pack(pady=10)

        tk.Label(bet_frame, text="Bet Amount:", font=('Arial', 12),
                 bg='#0B6623', fg='white').grid(row=0, column=0, padx=5)

        self.bet_entry = tk.Entry(bet_frame, font=('Arial', 12), width=10)
        self.bet_entry.grid(row=0, column=1, padx=5)

        # Betting buttons
        bet_buttons_frame = tk.Frame(self.root, bg='#0B6623')
        bet_buttons_frame.pack(pady=5)

        for amount in [10, 25, 50, 100]:
            btn = tk.Button(bet_buttons_frame, text=f"${amount}",
                            command=lambda a=amount: self.quick_bet(a),
                            font=('Arial', 10), width=6)
            btn.pack(side=tk.LEFT, padx=3)

        # Action buttons
        action_frame = tk.Frame(self.root, bg='#0B6623')
        action_frame.pack(pady=15)

        self.deal_button = tk.Button(action_frame, text="DEAL", command=self.deal_cards,
                                     font=('Arial', 12, 'bold'), width=10, bg='#4CAF50', fg='white')
        self.deal_button.grid(row=0, column=0, padx=5, pady=5)

        self.hit_button = tk.Button(action_frame, text="HIT", command=self.hit,
                                    font=('Arial', 12, 'bold'), width=10, bg='#2196F3', fg='white',
                                    state=tk.DISABLED)
        self.hit_button.grid(row=0, column=1, padx=5, pady=5)

        self.stand_button = tk.Button(action_frame, text="STAND", command=self.stand,
                                      font=('Arial', 12, 'bold'), width=10, bg='#FF9800', fg='white',
                                      state=tk.DISABLED)
        self.stand_button.grid(row=0, column=2, padx=5, pady=5)

        self.double_button = tk.Button(action_frame, text="DOUBLE", command=self.double_down,
                                       font=('Arial', 12, 'bold'), width=10, bg='#9C27B0', fg='white',
                                       state=tk.DISABLED)
        self.double_button.grid(row=1, column=0, padx=5, pady=5)

        self.split_button = tk.Button(action_frame, text="SPLIT", command=self.split,
                                      font=('Arial', 12, 'bold'), width=10, bg='#FF5722', fg='white',
                                      state=tk.DISABLED)
        self.split_button.grid(row=1, column=1, padx=5, pady=5)

        self.insurance_button = tk.Button(action_frame, text="INSURANCE", command=self.insurance,
                                          font=('Arial', 12, 'bold'), width=10, bg='#607D8B', fg='white',
                                          state=tk.DISABLED)
        self.insurance_button.grid(row=1, column=2, padx=5, pady=5)

        # Message label
        self.message_label = tk.Label(self.root, text="Place your bet and click DEAL",
                                      font=('Arial', 14, 'bold'), bg='#0B6623', fg='yellow')
        self.message_label.pack(pady=10)

    def quick_bet(self, amount):
        if not self.game_active:
            self.bet_entry.delete(0, tk.END)
            self.bet_entry.insert(0, str(amount))

    def deal_cards(self):
        try:
            bet = int(self.bet_entry.get())
            if bet <= 0:
                messagebox.showerror("Error", "Bet must be positive!")
                return
            if bet > self.balance:
                messagebox.showerror("Error", "Insufficient balance!")
                return
        except ValueError:
            messagebox.showerror("Error", "Enter a valid bet amount!")
            return

        self.current_bet = bet
        self.balance -= bet
        self.game_active = True
        self.insurance_bet = 0

        # Reset deck if running low
        if len(self.deck.cards) < 20:
            self.deck = Deck(6)

        # Initialize hands
        self.player_hands = [Hand()]
        self.player_hands[0].bet = bet
        self.current_hand_index = 0
        self.dealer_hand = Hand()

        # Deal initial cards
        for _ in range(2):
            self.player_hands[0].add_card(self.deck.deal())
            self.dealer_hand.add_card(self.deck.deal())

        self.update_display(hide_dealer_card=True)

        # Check for blackjack
        if self.player_hands[0].is_blackjack():
            if self.dealer_hand.cards[0].rank == 'A' or self.dealer_hand.cards[0].value() == 10:
                self.update_display(hide_dealer_card=False)
                if self.dealer_hand.is_blackjack():
                    self.end_game("Push! Both have Blackjack!", 0)
                else:
                    self.end_game("Blackjack! You win!", 2.5)
            else:
                self.end_game("Blackjack! You win!", 2.5)
            return

        # Check for insurance
        if self.dealer_hand.cards[0].rank == 'A':
            self.insurance_button.config(state=tk.NORMAL)

        # Enable action buttons
        self.deal_button.config(state=tk.DISABLED)
        self.hit_button.config(state=tk.NORMAL)
        self.stand_button.config(state=tk.NORMAL)

        if self.player_hands[0].can_double() and bet * 2 <= self.balance + bet:
            self.double_button.config(state=tk.NORMAL)

        if self.player_hands[0].can_split() and bet * 2 <= self.balance + bet:
            self.split_button.config(state=tk.NORMAL)

        self.message_label.config(text="Your turn!")

    def insurance(self):
        insurance_amount = self.current_bet // 2
        if insurance_amount > self.balance:
            messagebox.showerror("Error", "Insufficient balance for insurance!")
            return

        self.balance -= insurance_amount
        self.insurance_bet = insurance_amount
        self.insurance_button.config(state=tk.DISABLED)
        self.balance_label.config(text=f"Balance: ${self.balance}")
        self.message_label.config(text=f"Insurance bet: ${insurance_amount}")

    def hit(self):
        # Guard: ensure we have a valid hand to hit on
        if not self.player_hands or self.current_hand_index >= len(self.player_hands):
            return
        current_hand = self.player_hands[self.current_hand_index]
        current_hand.add_card(self.deck.deal())

        self.update_display(hide_dealer_card=True)

        # Disable double and split after first hit
        self.double_button.config(state=tk.DISABLED)
        self.split_button.config(state=tk.DISABLED)
        self.insurance_button.config(state=tk.DISABLED)

        if current_hand.is_busted():
            current_hand.is_finished = True
            self.message_label.config(text=f"Hand {self.current_hand_index + 1} busted!")
            self.next_hand()

    def stand(self):
        # Guard: ensure we have a valid hand to stand on
        if not self.player_hands or self.current_hand_index >= len(self.player_hands):
            return
        self.player_hands[self.current_hand_index].is_finished = True
        self.next_hand()

    def double_down(self):
        # Guard: ensure we have a valid hand to double on
        if not self.player_hands or self.current_hand_index >= len(self.player_hands):
            return
        current_hand = self.player_hands[self.current_hand_index]

        if current_hand.bet > self.balance:
            messagebox.showerror("Error", "Insufficient balance to double!")
            return

        self.balance -= current_hand.bet
        current_hand.bet *= 2
        current_hand.is_doubled = True
        current_hand.add_card(self.deck.deal())

        self.update_display(hide_dealer_card=True)
        self.balance_label.config(text=f"Balance: ${self.balance}")

        current_hand.is_finished = True

        if current_hand.is_busted():
            self.message_label.config(text=f"Hand {self.current_hand_index + 1} busted!")

        self.next_hand()

    def split(self):
        # Guard: ensure we have a valid hand to split
        if not self.player_hands or self.current_hand_index >= len(self.player_hands):
            return
        if self.current_bet > self.balance:
            messagebox.showerror("Error", "Insufficient balance to split!")
            return

        self.balance -= self.current_bet

        original_hand = self.player_hands[self.current_hand_index]
        new_hand = Hand()
        new_hand.bet = original_hand.bet
        new_hand.is_split = True
        original_hand.is_split = True

        # Move one card to new hand
        new_hand.add_card(original_hand.cards.pop())

        # Deal new cards
        original_hand.add_card(self.deck.deal())
        new_hand.add_card(self.deck.deal())

        self.player_hands.insert(self.current_hand_index + 1, new_hand)

        self.update_display(hide_dealer_card=True)
        self.balance_label.config(text=f"Balance: ${self.balance}")

        self.split_button.config(state=tk.DISABLED)
        self.double_button.config(
            state=tk.DISABLED if not self.player_hands[self.current_hand_index].can_double() else tk.NORMAL)

    def next_hand(self):
        self.current_hand_index += 1

        if self.current_hand_index < len(self.player_hands):
            self.update_display(hide_dealer_card=True)
            self.message_label.config(text=f"Playing hand {self.current_hand_index + 1}")

            current_hand = self.player_hands[self.current_hand_index]
            self.double_button.config(
                state=tk.NORMAL if current_hand.can_double() and current_hand.bet <= self.balance else tk.DISABLED)
        else:
            self.dealer_play()

    def dealer_play(self):
        self.update_display(hide_dealer_card=False)

        # Check insurance
        if self.insurance_bet > 0:
            if self.dealer_hand.is_blackjack():
                self.balance += self.insurance_bet * 3
                self.message_label.config(text=f"Dealer Blackjack! Insurance pays ${self.insurance_bet * 2}")
            else:
                self.message_label.config(text="Dealer doesn't have Blackjack. Insurance lost.")

        # Check if all player hands busted
        if all(hand.is_busted() for hand in self.player_hands):
            self.finish_game()
            return

        # Dealer draws
        self.root.after(500, self.dealer_draw_step)

    def dealer_draw_step(self):
        if self.dealer_hand.get_value() < 17:
            self.dealer_hand.add_card(self.deck.deal())
            self.update_display(hide_dealer_card=False)
            self.root.after(500, self.dealer_draw_step)
        else:
            self.finish_game()

    def finish_game(self):
        dealer_value = self.dealer_hand.get_value()
        results = []
        total_winnings = 0

        for i, hand in enumerate(self.player_hands):
            player_value = hand.get_value()

            if hand.is_busted():
                results.append(f"Hand {i + 1}: Lost")
            elif dealer_value > 21:
                winnings = hand.bet * 2
                total_winnings += winnings
                results.append(f"Hand {i + 1}: Won ${hand.bet}")
            elif player_value > dealer_value:
                winnings = hand.bet * 2
                total_winnings += winnings
                results.append(f"Hand {i + 1}: Won ${hand.bet}")
            elif player_value < dealer_value:
                results.append(f"Hand {i + 1}: Lost")
            else:
                total_winnings += hand.bet
                results.append(f"Hand {i + 1}: Push")

        self.balance += total_winnings
        self.balance_label.config(text=f"Balance: ${self.balance}")

        result_text = "\n".join(results)
        self.message_label.config(text=result_text)

        if self.balance <= 0:
            messagebox.showinfo("Game Over", "You're out of money!")
            self.balance = 1000
            self.balance_label.config(text=f"Balance: ${self.balance}")

        self.reset_game()

    def end_game(self, message, multiplier):
        winnings = int(self.current_bet * multiplier)
        self.balance += winnings
        self.balance_label.config(text=f"Balance: ${self.balance}")
        self.message_label.config(text=message)
        self.reset_game()

    def reset_game(self):
        self.game_active = False
        self.deal_button.config(state=tk.NORMAL)
        self.hit_button.config(state=tk.DISABLED)
        self.stand_button.config(state=tk.DISABLED)
        self.double_button.config(state=tk.DISABLED)
        self.split_button.config(state=tk.DISABLED)
        self.insurance_button.config(state=tk.DISABLED)

    def update_display(self, hide_dealer_card=False):
        # Update dealer display
        if hide_dealer_card:
            dealer_cards = f"{self.dealer_hand.cards[0]}  🂠"
            dealer_value = ""
        else:
            dealer_cards = "  ".join(str(card) for card in self.dealer_hand.cards)
            dealer_value = f"Value: {self.dealer_hand.get_value()}"
            if self.dealer_hand.is_blackjack():
                dealer_value += " (Blackjack!)"
            elif self.dealer_hand.is_busted():
                dealer_value += " (Bust!)"

        self.dealer_cards_label.config(text=dealer_cards)
        self.dealer_value_label.config(text=dealer_value)

        # Update player display
        if len(self.player_hands) == 1:
            player_cards = "  ".join(str(card) for card in self.player_hands[0].cards)
            player_value = f"Value: {self.player_hands[0].get_value()}"
            if self.player_hands[0].is_blackjack():
                player_value += " (Blackjack!)"
            elif self.player_hands[0].is_busted():
                player_value += " (Bust!)"
        else:
            hands_display = []
            for i, hand in enumerate(self.player_hands):
                cards = "  ".join(str(card) for card in hand.cards)
                marker = " ◄" if i == self.current_hand_index else ""
                hands_display.append(f"Hand {i + 1}{marker}: {cards}")
            player_cards = "\n".join(hands_display)

            current_hand = self.player_hands[self.current_hand_index]
            player_value = f"Hand {self.current_hand_index + 1} Value: {current_hand.get_value()}"
            if current_hand.is_busted():
                player_value += " (Bust!)"

        self.player_cards_label.config(text=player_cards)
        self.player_value_label.config(text=player_value)
        self.balance_label.config(text=f"Balance: ${self.balance}")


if __name__ == "__main__":
    root = tk.Tk()
    game = BlackjackGUI(root)
    root.mainloop()
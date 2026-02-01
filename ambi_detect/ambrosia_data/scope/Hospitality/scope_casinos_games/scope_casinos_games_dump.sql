BEGIN TRANSACTION;
CREATE TABLE Casinos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Casinos" VALUES(1,'Caesar Palace','Las Vegas');
INSERT INTO "Casinos" VALUES(2,'Bellagio','Las Vegas');
INSERT INTO "Casinos" VALUES(3,'Wynn Las Vegas','Las Vegas');
INSERT INTO "Casinos" VALUES(4,'The Venetian Resort Hotel Casino','Las Vegas');
INSERT INTO "Casinos" VALUES(5,'ARIA Resort & Casino','Las Vegas');
CREATE TABLE Casinos_Games (
    casino_id INTEGER,
    game_id INTEGER,
    FOREIGN KEY(casino_id) REFERENCES Casinos(id),
    FOREIGN KEY(game_id) REFERENCES Games(id),
    PRIMARY KEY(casino_id, game_id)
);
INSERT INTO "Casinos_Games" VALUES(1,2);
INSERT INTO "Casinos_Games" VALUES(1,3);
INSERT INTO "Casinos_Games" VALUES(1,4);
INSERT INTO "Casinos_Games" VALUES(1,5);
INSERT INTO "Casinos_Games" VALUES(2,1);
INSERT INTO "Casinos_Games" VALUES(2,3);
INSERT INTO "Casinos_Games" VALUES(2,4);
INSERT INTO "Casinos_Games" VALUES(2,5);
INSERT INTO "Casinos_Games" VALUES(3,1);
INSERT INTO "Casinos_Games" VALUES(3,2);
INSERT INTO "Casinos_Games" VALUES(3,4);
INSERT INTO "Casinos_Games" VALUES(3,5);
INSERT INTO "Casinos_Games" VALUES(4,1);
INSERT INTO "Casinos_Games" VALUES(4,2);
INSERT INTO "Casinos_Games" VALUES(4,3);
INSERT INTO "Casinos_Games" VALUES(4,5);
INSERT INTO "Casinos_Games" VALUES(5,1);
INSERT INTO "Casinos_Games" VALUES(5,2);
INSERT INTO "Casinos_Games" VALUES(5,3);
INSERT INTO "Casinos_Games" VALUES(5,4);
INSERT INTO "Casinos_Games" VALUES(5,5);
INSERT INTO "Casinos_Games" VALUES(1,1);
INSERT INTO "Casinos_Games" VALUES(2,2);
INSERT INTO "Casinos_Games" VALUES(3,3);
INSERT INTO "Casinos_Games" VALUES(4,4);
CREATE TABLE Games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_name TEXT,
    description TEXT
);
INSERT INTO "Games" VALUES(1,'Slot Machines','Electronic gaming machines that create a game of chance for their customers.');
INSERT INTO "Games" VALUES(2,'Blackjack','A comparing card game between one or more players and a dealer, where each player aims to beat the dealer by having cards with an aggregate value which is closer to 21 than those of the dealer without exceeding 21.');
INSERT INTO "Games" VALUES(3,'Roulette','A casino game named after the French word for little wheel. In the game, players may choose to place bets on either a single number, various groupings of numbers, the colors red or black, whether the total of all first and second-digit numbers is odd or even, or if the total of the first two digits is a multiple of a certain number such as 3.');
INSERT INTO "Games" VALUES(4,'Poker','Card game played with a standard deck of 52 playing cards, ranked from high to low, in which participants wager into a pot consisting of money contributed by the players themselves.');
INSERT INTO "Games" VALUES(5,'Craps','Dice game in which players make a series of wagers during the course of a round known as the roll. The main objective of most craps wagers is to correctly predict the outcome of the roll.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Casinos',5);
INSERT INTO "sqlite_sequence" VALUES('Games',5);
COMMIT;

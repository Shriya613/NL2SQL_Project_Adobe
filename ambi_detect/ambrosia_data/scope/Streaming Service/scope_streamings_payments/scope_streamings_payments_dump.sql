BEGIN TRANSACTION;
CREATE TABLE Genres (
    genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Genres" VALUES(1,'Action');
INSERT INTO "Genres" VALUES(2,'Comedy');
INSERT INTO "Genres" VALUES(3,'Drama');
INSERT INTO "Genres" VALUES(4,'Horror');
INSERT INTO "Genres" VALUES(5,'Romance');
CREATE TABLE Movies (
    movie_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    genre_id INTEGER,
    FOREIGN KEY(genre_id) REFERENCES Genres(genre_id)
);
INSERT INTO "Movies" VALUES(1,'Movie1',2021,1);
INSERT INTO "Movies" VALUES(2,'Movie2',2020,2);
INSERT INTO "Movies" VALUES(3,'Movie3',2019,3);
INSERT INTO "Movies" VALUES(4,'Movie4',2022,4);
INSERT INTO "Movies" VALUES(5,'Movie5',2018,5);
CREATE TABLE "Payments" (
    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    payment_type TEXT,
    FOREIGN KEY(payment_type) REFERENCES PaymentTypes(payment_type)
);
INSERT INTO "Payments" VALUES(1,'Credit_Card');
INSERT INTO "Payments" VALUES(2,'Debit_Card');
INSERT INTO "Payments" VALUES(3,'PayPal');
INSERT INTO "Payments" VALUES(4,'Bank_Transfer');
INSERT INTO "Payments" VALUES(5,'Google_Pay');
CREATE TABLE Streamings (
    streaming_id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id INTEGER,
    user_id INTEGER,
    stream_start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    stream_end_time TIMESTAMP,
    FOREIGN KEY(movie_id) REFERENCES Movies(movie_id),
    FOREIGN KEY(user_id) REFERENCES Users(id)
);
INSERT INTO "Streamings" VALUES(1,1,1,'2023-01-01 10:00:00','2023-01-01 12:00:00');
INSERT INTO "Streamings" VALUES(2,2,2,'2023-01-02 10:00:00','2023-01-02 12:00:00');
INSERT INTO "Streamings" VALUES(3,3,3,'2023-01-03 10:00:00','2023-01-03 12:00:00');
INSERT INTO "Streamings" VALUES(4,4,4,'2023-01-04 10:00:00','2023-01-04 12:00:00');
INSERT INTO "Streamings" VALUES(5,5,5,'2023-01-05 10:00:00','2023-01-05 12:00:00');
CREATE TABLE Streamings_Payments (
    streaming_id INTEGER,
    payment_id INTEGER,
    FOREIGN KEY(streaming_id) REFERENCES Streamings(streaming_id),
    FOREIGN KEY(payment_id) REFERENCES Payments(payment_id)
);
INSERT INTO "Streamings_Payments" VALUES(1,1);
INSERT INTO "Streamings_Payments" VALUES(2,1);
INSERT INTO "Streamings_Payments" VALUES(3,2);
INSERT INTO "Streamings_Payments" VALUES(4,2);
INSERT INTO "Streamings_Payments" VALUES(5,3);
INSERT INTO "Streamings_Payments" VALUES(3,1);
INSERT INTO "Streamings_Payments" VALUES(4,1);
INSERT INTO "Streamings_Payments" VALUES(5,1);
CREATE TABLE Subscriptions (
    subscription_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    start_date DATE,
    end_date DATE,
    status TEXT ,
    FOREIGN KEY(user_id) REFERENCES Users(id)
);
INSERT INTO "Subscriptions" VALUES(1,1,'2023-01-01','2023-12-31','Active');
INSERT INTO "Subscriptions" VALUES(2,2,'2023-02-01','2023-12-31','Inactive');
INSERT INTO "Subscriptions" VALUES(3,3,'2023-03-01','2023-12-31','Canceled');
INSERT INTO "Subscriptions" VALUES(4,4,'2023-04-01','2023-12-31','Expired');
INSERT INTO "Subscriptions" VALUES(5,5,'2023-05-01','2023-12-31','Active');
CREATE TABLE Users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    email TEXT
);
INSERT INTO "Users" VALUES(1,'User1','user1@example.com');
INSERT INTO "Users" VALUES(2,'User2','user2@example.com');
INSERT INTO "Users" VALUES(3,'User3','user3@example.com');
INSERT INTO "Users" VALUES(4,'User4','user4@example.com');
INSERT INTO "Users" VALUES(5,'User5','user5@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Users',5);
INSERT INTO "sqlite_sequence" VALUES('Genres',5);
INSERT INTO "sqlite_sequence" VALUES('Movies',5);
INSERT INTO "sqlite_sequence" VALUES('Subscriptions',5);
INSERT INTO "sqlite_sequence" VALUES('Streamings',5);
INSERT INTO "sqlite_sequence" VALUES('Payments',5);
COMMIT;

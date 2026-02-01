BEGIN TRANSACTION;
CREATE TABLE Captions_and_subtitles(
    caption_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT ,
    language TEXT
);
INSERT INTO "Captions_and_subtitles" VALUES(1,'Auto_generated_Subtitles','English');
INSERT INTO "Captions_and_subtitles" VALUES(2,'Manual_subtitles','Spanish');
INSERT INTO "Captions_and_subtitles" VALUES(3,'Auto_generated_Subtitles','French');
INSERT INTO "Captions_and_subtitles" VALUES(4,'Manual_subtitles','German');
INSERT INTO "Captions_and_subtitles" VALUES(5,'No_subtitles','Italian');
CREATE TABLE Payments (
    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount REAL,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "Payments" VALUES(1,9.99,'2024-01-30 18:45:30');
INSERT INTO "Payments" VALUES(2,19.99,'2024-01-30 18:45:30');
INSERT INTO "Payments" VALUES(3,29.99,'2024-01-30 18:45:30');
INSERT INTO "Payments" VALUES(4,39.99,'2024-01-30 18:45:30');
INSERT INTO "Payments" VALUES(5,49.99,'2024-01-30 18:45:30');
CREATE TABLE Streamings (
    streaming_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    duration REAL
);
INSERT INTO "Streamings" VALUES(1,'Movie1',300.0);
INSERT INTO "Streamings" VALUES(2,'TV Show1',200.0);
INSERT INTO "Streamings" VALUES(3,'Documentary1',400.0);
INSERT INTO "Streamings" VALUES(4,'Short Film1',60.0);
INSERT INTO "Streamings" VALUES(5,'Trailer1',90.0);
CREATE TABLE Streamings_captions_and_subtitles(
    streamings_id INTEGER,
    captions_and_subtitles_id INTEGER,
    FOREIGN KEY(streamings_id) REFERENCES Streamings(streaming_id),
    FOREIGN KEY(captions_and_subtitles_id) REFERENCES Captions_and_subtitles(caption_id)
);
INSERT INTO "Streamings_captions_and_subtitles" VALUES(1,1);
INSERT INTO "Streamings_captions_and_subtitles" VALUES(1,2);
INSERT INTO "Streamings_captions_and_subtitles" VALUES(2,3);
INSERT INTO "Streamings_captions_and_subtitles" VALUES(2,4);
INSERT INTO "Streamings_captions_and_subtitles" VALUES(3,5);
INSERT INTO "Streamings_captions_and_subtitles" VALUES(2,1);
INSERT INTO "Streamings_captions_and_subtitles" VALUES(3,1);
INSERT INTO "Streamings_captions_and_subtitles" VALUES(4,1);
INSERT INTO "Streamings_captions_and_subtitles" VALUES(5,1);
CREATE TABLE User_Payments (
    user_id INTEGER,
    payment_id INTEGER,
    FOREIGN KEY(user_id) REFERENCES Users(user_id),
    FOREIGN KEY(payment_id) REFERENCES Payments(payment_id)
);
INSERT INTO "User_Payments" VALUES(1,1);
INSERT INTO "User_Payments" VALUES(1,2);
INSERT INTO "User_Payments" VALUES(2,3);
INSERT INTO "User_Payments" VALUES(2,4);
INSERT INTO "User_Payments" VALUES(3,5);
CREATE TABLE Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    email TEXT
);
INSERT INTO "Users" VALUES(1,'User1','user1@example.com');
INSERT INTO "Users" VALUES(2,'User2','user2@example.com');
INSERT INTO "Users" VALUES(3,'User3','user3@example.com');
INSERT INTO "Users" VALUES(4,'User4','user4@example.com');
INSERT INTO "Users" VALUES(5,'User5','user5@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streamings',5);
INSERT INTO "sqlite_sequence" VALUES('Captions_and_subtitles',5);
INSERT INTO "sqlite_sequence" VALUES('Users',5);
INSERT INTO "sqlite_sequence" VALUES('Payments',5);
COMMIT;

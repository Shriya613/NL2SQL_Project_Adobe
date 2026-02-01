BEGIN TRANSACTION;
CREATE TABLE Episodes (
    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    show_id INTEGER,
    season INTEGER,
    episode_number INTEGER,
    title TEXT,
    duration INTEGER,
    release_date DATETIME,
    FOREIGN KEY(show_id) REFERENCES Shows(show_id)
);
INSERT INTO "Episodes" VALUES(1,1,1,1,'Episode 1',60,'2022-01-01');
INSERT INTO "Episodes" VALUES(2,2,1,1,'Episode 1',60,'2022-01-02');
CREATE TABLE Ratings (
    rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    show_id INTEGER,
    episode_id INTEGER,
    stars INTEGER ,
    comment TEXT,
    FOREIGN KEY(user_id) REFERENCES Users(user_id),
    FOREIGN KEY(show_id) REFERENCES Shows(show_id),
    FOREIGN KEY(episode_id) REFERENCES Episodes(episode_id)
);
INSERT INTO "Ratings" VALUES(1,1,1,1,5,'Great Show!');
INSERT INTO "Ratings" VALUES(2,2,2,1,4,'Funny but could be better.');
CREATE TABLE RealityShows (
    reality_show_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    network TEXT,
    host TEXT,
    premiere_date DATETIME,
    winner TEXT,
    city TEXT,
    country TEXT
);
INSERT INTO "RealityShows" VALUES(1,'RealityShow1','Network A','Host1','2022-01-01','Winner1','City1','Country1');
INSERT INTO "RealityShows" VALUES(2,'RealityShow2','Network B','Host2','2022-01-02','Winner2','City2','Country2');
CREATE TABLE Shows (
    show_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    genre TEXT,
    description TEXT,
    release_year INTEGER,
    language TEXT
);
INSERT INTO "Shows" VALUES(1,'Show1','Drama','A drama series about a detective',2020,'English');
INSERT INTO "Shows" VALUES(2,'Show2','Comedy','A comedy series about friends',2019,'English');
CREATE TABLE Subscriptions (
    subscription_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    plan TEXT,
    start_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_date DATETIME,
    FOREIGN KEY(user_id) REFERENCES Users(user_id)
);
INSERT INTO "Subscriptions" VALUES(1,1,'Basic','2022-01-01','2022-12-31');
INSERT INTO "Subscriptions" VALUES(2,2,'Premium','2022-01-02','2022-12-31');
CREATE TABLE Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    email TEXT,
    password TEXT,
    city TEXT,
    country TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "Users" VALUES(1,'User1','email@example.com','password123','New York','USA','2024-02-19 15:04:39');
INSERT INTO "Users" VALUES(2,'User2','email2@example.com','password321','Los Angeles','USA','2024-02-19 15:04:39');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Users',2);
INSERT INTO "sqlite_sequence" VALUES('Subscriptions',2);
INSERT INTO "sqlite_sequence" VALUES('Shows',2);
INSERT INTO "sqlite_sequence" VALUES('Episodes',2);
INSERT INTO "sqlite_sequence" VALUES('Ratings',2);
INSERT INTO "sqlite_sequence" VALUES('RealityShows',2);
COMMIT;

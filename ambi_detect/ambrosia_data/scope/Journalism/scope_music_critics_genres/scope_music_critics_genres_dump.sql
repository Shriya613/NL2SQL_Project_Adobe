BEGIN TRANSACTION;
CREATE TABLE Genres (
    genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255)
);
INSERT INTO "Genres" VALUES(1,'Jazz');
INSERT INTO "Genres" VALUES(2,'Rock');
INSERT INTO "Genres" VALUES(3,'Pop');
INSERT INTO "Genres" VALUES(4,'Classical');
INSERT INTO "Genres" VALUES(5,'Hip-hop');
CREATE TABLE Music_Critics (
    critic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Music_Critics" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Music_Critics" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Music_Critics" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Music_Critics" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Music_Critics" VALUES(5,'William Brown','william@example.com');
CREATE TABLE Music_Critics_Genres (
    music_critic_id INTEGER,
    genre_id INTEGER,
    FOREIGN KEY (music_critic_id) REFERENCES Music_Critics(critic_id),
    FOREIGN KEY (genre_id) REFERENCES Genres(genre_id),
    PRIMARY KEY (music_critic_id, genre_id)
);
INSERT INTO "Music_Critics_Genres" VALUES(1,1);
INSERT INTO "Music_Critics_Genres" VALUES(1,2);
INSERT INTO "Music_Critics_Genres" VALUES(2,3);
INSERT INTO "Music_Critics_Genres" VALUES(2,4);
INSERT INTO "Music_Critics_Genres" VALUES(3,5);
INSERT INTO "Music_Critics_Genres" VALUES(3,1);
INSERT INTO "Music_Critics_Genres" VALUES(4,3);
INSERT INTO "Music_Critics_Genres" VALUES(4,2);
INSERT INTO "Music_Critics_Genres" VALUES(5,4);
INSERT INTO "Music_Critics_Genres" VALUES(5,1);
INSERT INTO "Music_Critics_Genres" VALUES(2,1);
INSERT INTO "Music_Critics_Genres" VALUES(4,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Music_Critics',5);
INSERT INTO "sqlite_sequence" VALUES('Genres',5);
COMMIT;

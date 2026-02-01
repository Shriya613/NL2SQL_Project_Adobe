BEGIN TRANSACTION;
CREATE TABLE EpisodeGuests(
    id INTEGER PRIMARY KEY,
    episode_id INTEGER,
    guest_id INTEGER,
    role TEXT,
    appearance_duration INTEGER,
    FOREIGN KEY(episode_id) REFERENCES Episodes(id),
    FOREIGN KEY(guest_id) REFERENCES Guests(id)
);
INSERT INTO "EpisodeGuests" VALUES(1,1,1,'Interviewee',25);
INSERT INTO "EpisodeGuests" VALUES(2,1,2,'Panelist',25);
CREATE TABLE Episodes(
    id INTEGER PRIMARY KEY,
    show_id INTEGER,
    episode_number INTEGER,
    release_date DATE,
    duration INTEGER,
    description TEXT,
    FOREIGN KEY(show_id) REFERENCES Shows(id)
);
INSERT INTO "Episodes" VALUES(1,1,1,'2023-01-01',60,'Episode about comedy.');
INSERT INTO "Episodes" VALUES(2,2,1,'2023-01-08',60,'Episode about coffee.');
CREATE TABLE Guests(
    id INTEGER PRIMARY KEY,
    full_name TEXT,
    occupation TEXT,
    guest_on INTEGER,
    appearance_episode_number INTEGER,
    FOREIGN KEY(guest_on) REFERENCES Shows(title)
);
INSERT INTO "Guests" VALUES(1,'John Smith','Actor','The Alex Johnson Show',1);
INSERT INTO "Guests" VALUES(2,'Emily Davis','Author','Laughing Matters',1);
CREATE TABLE Podcasters(
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT,
    years_experience INTEGER
);
INSERT INTO "Podcasters" VALUES(1,'Alex Johnson','New York',10);
INSERT INTO "Podcasters" VALUES(2,'Jane Doe','Los Angeles',8);
CREATE TABLE Reviews(
    id INTEGER PRIMARY KEY,
    reviewer_name TEXT,
    rating INTEGER ,
    comments TEXT,
    show_id INTEGER,
    review_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(show_id) REFERENCES Shows(id)
);
INSERT INTO "Reviews" VALUES(1,'Alice Brown',5,'Great podcast!',1,'2024-03-07 21:13:47');
INSERT INTO "Reviews" VALUES(2,'Bob White',4,'Funny episodes.',2,'2024-03-07 21:13:47');
CREATE TABLE Shows(
    id INTEGER PRIMARY KEY,
    title TEXT,
    genre TEXT,
    network TEXT,
    audio_format TEXT ,
    host TEXT,
    FOREIGN KEY(host) REFERENCES Podcasters(name)
);
INSERT INTO "Shows" VALUES(1,'The Alex Johnson Show','Comedy','ABC','Podcast','Alex Johnson');
INSERT INTO "Shows" VALUES(2,'Laughing Matters','Comedy','NBC','Podcast','Jane Doe');
INSERT INTO "Shows" VALUES(3,'The Morning Rush','Talk Show','CBS Radio','Radio Show','Alex Johnson');
INSERT INTO "Shows" VALUES(4,'Morning Coffee','Talk Show','Fox Radio','Radio Show','Jane Doe');
COMMIT;

BEGIN TRANSACTION;
CREATE TABLE Character (
    id INTEGER PRIMARY KEY,
    name TEXT,
    role TEXT,
    story_id INTEGER,
    FOREIGN KEY(story_id) REFERENCES Story(id)
);
INSERT INTO "Character" VALUES(1,'Inspector Lestrade','Detective',1);
INSERT INTO "Character" VALUES(2,'Al Capone','Mob Boss',2);
CREATE TABLE Episode (
    id INTEGER PRIMARY KEY,
    episode_number INTEGER,
    title TEXT,
    release_date DATE,
    story_id INTEGER,
    FOREIGN KEY(story_id) REFERENCES Story(id)
);
INSERT INTO "Episode" VALUES(1,1,'Episode One - The Murder','2022-06-01',1);
INSERT INTO "Episode" VALUES(2,2,'Episode Two - The Chase','2022-06-08',1);
CREATE TABLE Review (
    id INTEGER PRIMARY KEY,
    reviewer_name TEXT,
    rating INTEGER ,
    comment TEXT,
    episode_id INTEGER,
    FOREIGN KEY(episode_id) REFERENCES Episode(id)
);
INSERT INTO "Review" VALUES(1,'John Doe',5,'Great acting!',1);
INSERT INTO "Review" VALUES(2,'Jane Smith',4,'Interesting plot but could be better',2);
CREATE TABLE "Setting" (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);
INSERT INTO "Setting" VALUES(1,'New York','The city that never sleeps');
INSERT INTO "Setting" VALUES(2,'London','Home of Big Ben and Buckingham Palace');
CREATE TABLE "Story" (
    id INTEGER PRIMARY KEY,
    title TEXT,
    location_id INTEGER,
    FOREIGN KEY(location_id) REFERENCES "Setting"(id));
INSERT INTO "Story" VALUES(1,'A Victorian Mystery',1);
INSERT INTO "Story" VALUES(2,'Gangsters of the Roaring Twenties',2);
COMMIT;

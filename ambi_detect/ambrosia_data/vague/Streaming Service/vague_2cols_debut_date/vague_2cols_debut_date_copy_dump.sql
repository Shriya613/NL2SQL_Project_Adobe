BEGIN TRANSACTION;
CREATE TABLE Category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE);
INSERT INTO "Category" VALUES(1,'Entertainment');
INSERT INTO "Category" VALUES(2,'Technology');
INSERT INTO "Category" VALUES(3,'Food & Cooking');
INSERT INTO "Category" VALUES(4,'Health & Fitness');
INSERT INTO "Category" VALUES(5,'News');
CREATE TABLE Episode (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Podcast_id INTEGER,
    Episode_Number INTEGER,
    Release_Date TEXT,
    Length INTEGER,
    FOREIGN KEY(Podcast_id) REFERENCES Podcast(id));
INSERT INTO "Episode" VALUES(1,1,1,'2021-07-01',60);
INSERT INTO "Episode" VALUES(2,1,2,'2021-07-08',60);
INSERT INTO "Episode" VALUES(3,2,1,'2021-07-15',30);
INSERT INTO "Episode" VALUES(4,3,1,'2021-07-01',45);
INSERT INTO "Episode" VALUES(5,4,1,'2020-07-01',35);
CREATE TABLE "Podcast" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Debut_Date TEXT,
    Title TEXT,
    Hosts TEXT,
    Duration INTEGER);
INSERT INTO "Podcast" VALUES(1,'Monday','The Morning Show','Jane Doe and John Smith',60);
INSERT INTO "Podcast" VALUES(2,'Tuesday','Technology Today','Alice Johnson',30);
INSERT INTO "Podcast" VALUES(3,'Wednesday','Cooking with Chef Mike','Mike Brown',45);
INSERT INTO "Podcast" VALUES(4,'Thursday','Health Matters','Dr. Lisa Wilson',35);
INSERT INTO "Podcast" VALUES(5,'Friday','News Roundup','Tom Harris',60);
CREATE TABLE Podcast_Category (
    Podcast_id INTEGER,
    Category_id INTEGER,
    PRIMARY KEY (Podcast_id, Category_id),
    FOREIGN KEY(Podcast_id) REFERENCES Podcast(id),
    FOREIGN KEY(Category_id) REFERENCES Category(id));
INSERT INTO "Podcast_Category" VALUES(1,1);
INSERT INTO "Podcast_Category" VALUES(2,2);
INSERT INTO "Podcast_Category" VALUES(3,3);
INSERT INTO "Podcast_Category" VALUES(4,4);
INSERT INTO "Podcast_Category" VALUES(5,5);
CREATE TABLE Subscriber (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Email TEXT,
    Password TEXT,
    Joined_Date TEXT);
INSERT INTO "Subscriber" VALUES(1,'john@example.com','password123','2021-06-01');
INSERT INTO "Subscriber" VALUES(2,'alice@example.com','secret456','2021-06-15');
INSERT INTO "Subscriber" VALUES(3,'bob@example.com','abc789','2021-07-01');
INSERT INTO "Subscriber" VALUES(4,'charlie@example.com','xyz123','2021-07-15');
INSERT INTO "Subscriber" VALUES(5,'diana@example.com','mno456','2021-08-01');
CREATE TABLE Subscription (
    Subscriber_id INTEGER,
    Podcast_id INTEGER,
    Start_Date TEXT,
    End_Date TEXT,
    PRIMARY KEY (Subscriber_id, Podcast_id),
    FOREIGN KEY(Subscriber_id) REFERENCES Subscriber(id),
    FOREIGN KEY(Podcast_id) REFERENCES Podcast(id));
INSERT INTO "Subscription" VALUES(1,1,'2021-07-01','2021-08-01');
INSERT INTO "Subscription" VALUES(1,2,'2021-07-01','2021-08-01');
INSERT INTO "Subscription" VALUES(2,3,'2021-07-15','2021-08-15');
INSERT INTO "Subscription" VALUES(3,4,'2021-07-01','2021-08-01');
INSERT INTO "Subscription" VALUES(4,5,'2021-07-15','2021-08-15');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Episode',5);
INSERT INTO "sqlite_sequence" VALUES('Category',5);
INSERT INTO "sqlite_sequence" VALUES('Subscriber',5);
INSERT INTO "sqlite_sequence" VALUES('Podcast',5);
COMMIT;

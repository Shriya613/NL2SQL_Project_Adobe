BEGIN TRANSACTION;
CREATE TABLE Acts(
    act_id INTEGER PRIMARY KEY,
    act_name TEXT
);
INSERT INTO "Acts" VALUES(1,'Headliner');
INSERT INTO "Acts" VALUES(2,'Opening Act');
INSERT INTO "Acts" VALUES(3,'Special Guest');
INSERT INTO "Acts" VALUES(4,'Supporting Act');
INSERT INTO "Acts" VALUES(5,'Featured Artist');
CREATE TABLE Comedy_Shows(
    comedy_show_id INTEGER PRIMARY KEY,
    show_title TEXT,
    date DATETIME,
    location TEXT
);
INSERT INTO "Comedy_Shows" VALUES(1,'Laugh Out Loud','2022-09-01 19:30:00','New York City');
INSERT INTO "Comedy_Shows" VALUES(2,'Humor Festival','2022-09-15 18:00:00','Los Angeles');
INSERT INTO "Comedy_Shows" VALUES(3,'Giggle Gala','2022-10-01 20:00:00','Chicago');
INSERT INTO "Comedy_Shows" VALUES(4,'Wit & Wisdom','2022-10-15 17:00:00','Boston');
INSERT INTO "Comedy_Shows" VALUES(5,'Side Splitter','2022-11-01 21:00:00','Miami');
CREATE TABLE Comedy_Shows_Acts(
    acts_comedy_shows_id INTEGER,
    comedy_show_id INTEGER,
    FOREIGN KEY(acts_comedy_shows_id) REFERENCES Acts(act_id),
    FOREIGN KEY(comedy_show_id) REFERENCES Comedy_Shows(comedy_show_id)
);
INSERT INTO "Comedy_Shows_Acts" VALUES(1,1);
INSERT INTO "Comedy_Shows_Acts" VALUES(2,1);
INSERT INTO "Comedy_Shows_Acts" VALUES(3,2);
INSERT INTO "Comedy_Shows_Acts" VALUES(4,2);
INSERT INTO "Comedy_Shows_Acts" VALUES(5,3);
INSERT INTO "Comedy_Shows_Acts" VALUES(6,3);
INSERT INTO "Comedy_Shows_Acts" VALUES(7,4);
INSERT INTO "Comedy_Shows_Acts" VALUES(8,4);
INSERT INTO "Comedy_Shows_Acts" VALUES(9,5);
INSERT INTO "Comedy_Shows_Acts" VALUES(10,5);
INSERT INTO "Comedy_Shows_Acts" VALUES(1,2);
INSERT INTO "Comedy_Shows_Acts" VALUES(1,3);
INSERT INTO "Comedy_Shows_Acts" VALUES(1,4);
INSERT INTO "Comedy_Shows_Acts" VALUES(1,5);
COMMIT;

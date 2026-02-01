BEGIN TRANSACTION;
CREATE TABLE Content_Promotions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    description TEXT);
INSERT INTO "Content_Promotions" VALUES(1,'Cross_promotional_Partnerships','This promotion crosses over between our different platforms.');
INSERT INTO "Content_Promotions" VALUES(2,'Discounted Subscription','Get a discount on your subscription by watching this content.');
INSERT INTO "Content_Promotions" VALUES(3,'Bundled Content','Purchase one item and get another at a reduced price.');
INSERT INTO "Content_Promotions" VALUES(4,'Exclusive Access','Unlock exclusive access to special features when you watch this content.');
INSERT INTO "Content_Promotions" VALUES(5,'Referral Program Reward','Earn rewards from our referral program by watching this content.');
CREATE TABLE Streamings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE);
INSERT INTO "Streamings" VALUES(1,'Netflix','2019-06-30');
INSERT INTO "Streamings" VALUES(2,'Amazon Prime','2020-07-01');
INSERT INTO "Streamings" VALUES(3,'BBC iPlayer','2021-08-02');
INSERT INTO "Streamings" VALUES(4,'Disney+','2022-09-03');
INSERT INTO "Streamings" VALUES(5,'AppleTV','2023-10-04');
CREATE TABLE Streamings_Content_Promotions(
    streaming_id INTEGER,
    content_promotion_type_id INTEGER,
    FOREIGN KEY (streaming_id) REFERENCES Streamings(id),
    FOREIGN KEY (content_promotion_type_id) REFERENCES Content_Promotions(id));
INSERT INTO "Streamings_Content_Promotions" VALUES(1,1);
INSERT INTO "Streamings_Content_Promotions" VALUES(2,2);
INSERT INTO "Streamings_Content_Promotions" VALUES(3,3);
INSERT INTO "Streamings_Content_Promotions" VALUES(4,4);
INSERT INTO "Streamings_Content_Promotions" VALUES(5,5);
INSERT INTO "Streamings_Content_Promotions" VALUES(2,1);
INSERT INTO "Streamings_Content_Promotions" VALUES(3,1);
INSERT INTO "Streamings_Content_Promotions" VALUES(4,1);
INSERT INTO "Streamings_Content_Promotions" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streamings',5);
INSERT INTO "sqlite_sequence" VALUES('Content_Promotions',5);
COMMIT;

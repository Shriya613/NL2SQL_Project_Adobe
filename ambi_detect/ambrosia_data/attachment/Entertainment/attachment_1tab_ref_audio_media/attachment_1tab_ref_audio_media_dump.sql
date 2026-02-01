BEGIN TRANSACTION;
CREATE TABLE Audio_Media(
    MediaID INTEGER PRIMARY KEY,
    Title TEXT,
    Category TEXT,
    ReleaseDate DATE,
    Creator TEXT);
INSERT INTO "Audio_Media" VALUES(1,'The Daily','Podcast','2018-03-07','New York Times');
INSERT INTO "Audio_Media" VALUES(2,'TED Talks Daily','Podcast','2019-06-18','TED Conferences LLC');
INSERT INTO "Audio_Media" VALUES(3,'The Breakfast Club','Radio Show','2018-06-04','Power 105.1 FM');
INSERT INTO "Audio_Media" VALUES(4,'Morning Edition','Radio Show','2018-02-06','NPR');
CREATE TABLE Episode_Details(
    EpisodeID INTEGER PRIMARY KEY,
    AudioMediaID INTEGER,
    Title TEXT,
    Length INTEGER,
    FOREIGN KEY(AudioMediaID) REFERENCES Audio_Media(MediaID));
INSERT INTO "Episode_Details" VALUES(1,1,'Daily News Update',30);
INSERT INTO "Episode_Details" VALUES(2,1,'Politics & World Events',30);
INSERT INTO "Episode_Details" VALUES(3,2,'Inspirational Stories',45);
INSERT INTO "Episode_Details" VALUES(4,2,'Technology & Science',45);
INSERT INTO "Episode_Details" VALUES(5,3,'Interviews with Celebrities',60);
INSERT INTO "Episode_Details" VALUES(6,3,'News and Current Affairs',60);
INSERT INTO "Episode_Details" VALUES(7,4,'Feature Story on Education',45);
INSERT INTO "Episode_Details" VALUES(8,4,'Arts and Culture Segment',45);
CREATE TABLE Other_Category(
    CategoryID INTEGER PRIMARY KEY,
    Name TEXT);
INSERT INTO "Other_Category" VALUES(1,'Literature');
INSERT INTO "Other_Category" VALUES(2,'History');
INSERT INTO "Other_Category" VALUES(3,'Art');
INSERT INTO "Other_Category" VALUES(4,'Travel');
INSERT INTO "Other_Category" VALUES(5,'Fashion');
CREATE TABLE Podcast_Category(
    CategoryID INTEGER PRIMARY KEY,
    Name TEXT);
INSERT INTO "Podcast_Category" VALUES(1,'Business');
INSERT INTO "Podcast_Category" VALUES(2,'Health');
INSERT INTO "Podcast_Category" VALUES(3,'Technology');
INSERT INTO "Podcast_Category" VALUES(4,'Entertainment');
INSERT INTO "Podcast_Category" VALUES(5,'Comedy');
CREATE TABLE RadioShow_Category(
    CategoryID INTEGER PRIMARY KEY,
    Name TEXT);
INSERT INTO "RadioShow_Category" VALUES(1,'Music');
INSERT INTO "RadioShow_Category" VALUES(2,'Sports');
INSERT INTO "RadioShow_Category" VALUES(3,'News');
INSERT INTO "RadioShow_Category" VALUES(4,'Talk Shows');
INSERT INTO "RadioShow_Category" VALUES(5,'Children''s Programming');
COMMIT;

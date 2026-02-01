BEGIN TRANSACTION;
CREATE TABLE Categories(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT
);
INSERT INTO "Categories" VALUES(1,'Breaking News');
INSERT INTO "Categories" VALUES(2,'Politics');
INSERT INTO "Categories" VALUES(3,'Economy');
INSERT INTO "Categories" VALUES(4,'Technology');
INSERT INTO "Categories" VALUES(5,'Entertainment');
CREATE TABLE Online_News_Portals(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    website TEXT
);
INSERT INTO "Online_News_Portals" VALUES(1,'New York Times','https://www.nytimes.com/');
INSERT INTO "Online_News_Portals" VALUES(2,'The Guardian','https://www.theguardian.com/');
INSERT INTO "Online_News_Portals" VALUES(3,'BBC News','https://www.bbc.co.uk/news');
INSERT INTO "Online_News_Portals" VALUES(4,'Al Jazeera','https://www.aljazeera.com/');
INSERT INTO "Online_News_Portals" VALUES(5,'Deutsche Welle','https://www.dw.com/');
CREATE TABLE Online_News_Portals_Categories(
    online_news_portal_id INTEGER,
    category_id INTEGER,
    FOREIGN KEY(online_news_portal_id) REFERENCES Online_News_Portals(id),
    FOREIGN KEY(category_id) REFERENCES Categories(id)
);
INSERT INTO "Online_News_Portals_Categories" VALUES(1,1);
INSERT INTO "Online_News_Portals_Categories" VALUES(2,1);
INSERT INTO "Online_News_Portals_Categories" VALUES(3,1);
INSERT INTO "Online_News_Portals_Categories" VALUES(4,1);
INSERT INTO "Online_News_Portals_Categories" VALUES(5,1);
INSERT INTO "Online_News_Portals_Categories" VALUES(5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Categories',5);
INSERT INTO "sqlite_sequence" VALUES('Online_News_Portals',5);
COMMIT;

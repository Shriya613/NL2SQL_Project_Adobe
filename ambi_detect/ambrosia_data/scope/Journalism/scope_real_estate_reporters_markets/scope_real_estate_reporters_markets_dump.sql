BEGIN TRANSACTION;
CREATE TABLE Articles(
    article_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    published_date DATE
);
INSERT INTO "Articles" VALUES(1,'New Luxury Home Development','Details about a new luxury home development...','2021-09-01');
INSERT INTO "Articles" VALUES(2,'First Time Buyers Guide','Tips for first time home buyers...','2021-08-15');
INSERT INTO "Articles" VALUES(3,'Office Space Trends','The latest trends in office space...','2021-07-30');
INSERT INTO "Articles" VALUES(4,'Farmland vs Investment Property','Comparing farmland and investment property...','2021-06-20');
INSERT INTO "Articles" VALUES(5,'Retail Space Revival','How retail spaces are making a comeback...','2021-05-15');
CREATE TABLE Markets(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_name TEXT,
    description TEXT
);
INSERT INTO "Markets" VALUES(1,'Luxury Homes','High-end residential properties.');
INSERT INTO "Markets" VALUES(2,'Affordable Housing','Moderately priced homes for first-time buyers.');
INSERT INTO "Markets" VALUES(3,'Commercial Properties','Offices, retail spaces, industrial buildings.');
INSERT INTO "Markets" VALUES(4,'Rural Land','Agriculture land and rural estates.');
INSERT INTO "Markets" VALUES(5,'Investment Properties','Properties purchased for investment purposes.');
CREATE TABLE Real_Estate_Reporters(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Real_Estate_Reporters" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Real_Estate_Reporters" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Real_Estate_Reporters" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Real_Estate_Reporters" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Real_Estate_Reporters" VALUES(5,'David Brown','david@example.com');
CREATE TABLE Real_Estate_Reporters_Markets(
    reporter_id INTEGER,
    market_id INTEGER,
    FOREIGN KEY(reporter_id) REFERENCES Real_Estate_Reporters(id),
    FOREIGN KEY(market_id) REFERENCES Markets(id)
);
INSERT INTO "Real_Estate_Reporters_Markets" VALUES(1,1);
INSERT INTO "Real_Estate_Reporters_Markets" VALUES(2,1);
INSERT INTO "Real_Estate_Reporters_Markets" VALUES(3,1);
INSERT INTO "Real_Estate_Reporters_Markets" VALUES(4,1);
INSERT INTO "Real_Estate_Reporters_Markets" VALUES(5,1);
INSERT INTO "Real_Estate_Reporters_Markets" VALUES(5,2);
CREATE TABLE Reporter_Articles(
    reporter_id INTEGER,
    article_id INTEGER,
    FOREIGN KEY(reporter_id) REFERENCES Real_Estate_Reporters(id),
    FOREIGN KEY(article_id) REFERENCES Articles(article_id)
);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Real_Estate_Reporters',5);
INSERT INTO "sqlite_sequence" VALUES('Markets',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
COMMIT;

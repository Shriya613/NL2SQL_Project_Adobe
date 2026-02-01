BEGIN TRANSACTION;
CREATE TABLE MagazineArticles(
    ArticleID INTEGER PRIMARY KEY,
    Title TEXT,
    Category TEXT,
    Writer TEXT,
    IssueDate DATE);
INSERT INTO "MagazineArticles" VALUES(1,'The Future of AI','Tech Trends','Elizabeth Taylor','2023-06-01');
INSERT INTO "MagazineArticles" VALUES(2,'Top 10 Travel Destinations','Explore World','Mark Thompson','2023-07-01');
INSERT INTO "MagazineArticles" VALUES(3,'How to Cook the Perfect Steak','Kitchen Secrets','Rachel Green','2023-08-01');
INSERT INTO "MagazineArticles" VALUES(4,'Investment Tips for Beginners','Financial Advice','George Bailey','2023-09-01');
INSERT INTO "MagazineArticles" VALUES(5,'Understanding Climate Change','Environment Today','Simon Cowell','2023-10-01');
CREATE TABLE NewsReports(
    ReportID INTEGER PRIMARY KEY,
    EventName TEXT,
    Reporter TEXT,
    Location TEXT,
    DatePublished DATE);
INSERT INTO "NewsReports" VALUES(1,'Government Policy Change','Alice Cooper','Washington DC','2023-01-01');
INSERT INTO "NewsReports" VALUES(2,'Natural Disaster','Bob Lee','California','2023-02-01');
INSERT INTO "NewsReports" VALUES(3,'Business Merger','Charlie Adams','New York City','2023-03-01');
INSERT INTO "NewsReports" VALUES(4,'Political Scandal','Diana King','London','2023-04-01');
INSERT INTO "NewsReports" VALUES(5,'War Conflict','Edward Jones','Ukraine','2023-05-01');
CREATE TABLE "Personal_Essays"(
    EssayID INTEGER PRIMARY KEY, Title TEXT,
    Author TEXT,
    Topic TEXT,
    Length INTEGER,
    PublishDate DATE);
INSERT INTO "Personal_Essays" VALUES(1,'Wanderlust: A Traveler’s Guide','Evelyn O''Connor','Travel',700,'2023-06-01');
INSERT INTO "Personal_Essays" VALUES(2,'Family Ties: Connections that Bind','Jane Doe','Family',500,'2023-07-01');
INSERT INTO "Personal_Essays" VALUES(3,'Navigating the Tech Tides','William Jackson','Technology',800,'2023-08-01');
INSERT INTO "Personal_Essays" VALUES(4,'Secrets of the Kitchen','Evelyn O''Connor','Cooking',600,'2023-09-01');
INSERT INTO "Personal_Essays" VALUES(5,'Pathways to Health','Olivia Wilson','Health',900,'2023-10-01');
CREATE TABLE "Short_Stories"(
    StoryID INTEGER PRIMARY KEY, Title TEXT,
    Author TEXT,
    Genre TEXT,
    WordCount INTEGER,
    PublicationDate DATE);
INSERT INTO "Short_Stories" VALUES(1,'The Quiet Village','Evelyn O''Connor','Fiction',2000,'2023-01-01');
INSERT INTO "Short_Stories" VALUES(2,'The Mystery House','John Smith','Mystery',1800,'2023-02-01');
INSERT INTO "Short_Stories" VALUES(3,'Love in the Rain','Emily Johnson','Romance',1900,'2023-03-01');
INSERT INTO "Short_Stories" VALUES(4,'The Lost Treasure','David Brown','Adventure',2100,'2023-04-01');
INSERT INTO "Short_Stories" VALUES(5,'Truth Unveiled','Evelyn O''Connor','Nonfiction',2200,'2023-05-01');
COMMIT;

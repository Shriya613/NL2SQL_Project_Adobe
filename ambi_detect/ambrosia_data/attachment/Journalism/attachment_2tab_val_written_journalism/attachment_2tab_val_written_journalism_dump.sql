BEGIN TRANSACTION;
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    Country VARCHAR(255)
);
INSERT INTO "Authors" VALUES(1,'John','Doe','1980-01-01','USA');
INSERT INTO "Authors" VALUES(2,'Jane','Smith','1978-06-30','UK');
INSERT INTO "Authors" VALUES(3,'Emily','Brown','1992-11-20','Canada');
INSERT INTO "Authors" VALUES(4,'Michael','White','1990-07-15','Australia');
INSERT INTO "Authors" VALUES(5,'Rachel','Green','1985-03-25','Germany');
CREATE TABLE Feature_Articles(
    ArticleID INTEGER PRIMARY KEY,
    Title TEXT,
    Content TEXT,
    Word_Count INT,
    Published_Date TIMESTAMP,
    AuthorID INTEGER,
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID)
);
INSERT INTO "Feature_Articles" VALUES(1,'Travel Adventures','An adventurous journey through South America...',1000,'2022-01-01',1);
INSERT INTO "Feature_Articles" VALUES(2,'Culinary Delights','Exploring unique cuisine across Europe...',900,'2022-01-10',2);
INSERT INTO "Feature_Articles" VALUES(3,'Nature Exploration','Discovering hidden gems of nature around the globe...',1100,'2022-01-15',3);
INSERT INTO "Feature_Articles" VALUES(4,'Historical Treasures','Unraveling the mysteries of ancient civilizations...',1200,'2022-01-20',4);
INSERT INTO "Feature_Articles" VALUES(5,'Fashion Trends','A look at the latest fashion trends worldwide...',800,'2022-01-25',5);
CREATE TABLE News_Articles(
    ArticleID INTEGER PRIMARY KEY,
    Title TEXT,
    Content TEXT,
    Word_Count INT,
    Published_Date TIMESTAMP,
    AuthorID INTEGER,
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID)
);
INSERT INTO "News_Articles" VALUES(1,'Global Warming','The effects of global warming on marine life...',1000,'2022-01-01',1);
INSERT INTO "News_Articles" VALUES(2,'New Technology','A breakthrough in renewable energy technology...',500,'2022-01-10',2);
INSERT INTO "News_Articles" VALUES(3,'Politics Today','The current political landscape around the world...',800,'2022-01-15',3);
INSERT INTO "News_Articles" VALUES(4,'Economic Growth','Analysis of economic growth trends worldwide...',1200,'2022-01-20',4);
INSERT INTO "News_Articles" VALUES(5,'Health Crisis','The ongoing health crisis and its impact on society.',700,'2022-01-25',5);
CREATE TABLE Publications(
    PublisherID INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    Website URL
);
INSERT INTO "Publications" VALUES(1,'World Times','123 Main St, New York, NY','https://www.worldtimes.com');
INSERT INTO "Publications" VALUES(2,'European Gazette','456 London Rd, London, UK','https://europeangazette.eu');
INSERT INTO "Publications" VALUES(3,'Asian Insider','789 Tokyo Blvd, Tokyo, Japan','https://asianinsider.asia');
INSERT INTO "Publications" VALUES(4,'South American Review','123 Santiago Ave, Santiago, Chile','https://southamericanreview.cl');
INSERT INTO "Publications" VALUES(5,'Oceanic Observer','456 Sydney Cove, Sydney, Australia','https://oceanicobserver.au');
CREATE TABLE Publishers_and_Authors(
    PublisherID INTEGER,
    AuthorID INTEGER,
    FOREIGN KEY (PublisherID) REFERENCES Publications(PublisherID),
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID),
    PRIMARY KEY (PublisherID, AuthorID)
);
INSERT INTO "Publishers_and_Authors" VALUES(1,1);
INSERT INTO "Publishers_and_Authors" VALUES(2,2);
INSERT INTO "Publishers_and_Authors" VALUES(3,3);
INSERT INTO "Publishers_and_Authors" VALUES(4,4);
INSERT INTO "Publishers_and_Authors" VALUES(5,5);
COMMIT;

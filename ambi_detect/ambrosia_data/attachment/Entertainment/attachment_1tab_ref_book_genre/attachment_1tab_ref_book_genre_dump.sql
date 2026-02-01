BEGIN TRANSACTION;
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Country TEXT
);
INSERT INTO "Authors" VALUES(1,'John Doe','USA');
INSERT INTO "Authors" VALUES(2,'Jane Smith','UK');
INSERT INTO "Authors" VALUES(3,'Alice Johnson','Canada');
INSERT INTO "Authors" VALUES(4,'Bob Brown','Australia');
INSERT INTO "Authors" VALUES(5,'Charlie White','Germany');
CREATE TABLE BookPublishingHistory(
    PublishingID INTEGER PRIMARY KEY AUTOINCREMENT,
    ISBN VARCHAR(13),
    PublisherID INTEGER,
    YearOfPublication INT,
    FOREIGN KEY(ISBN) REFERENCES Books(ISBN),
    FOREIGN KEY(PublisherID) REFERENCES Publishers(PublisherID)
);
INSERT INTO "BookPublishingHistory" VALUES(1,'1234567890123',1,2018);
INSERT INTO "BookPublishingHistory" VALUES(2,'9876543210987',2,2020);
INSERT INTO "BookPublishingHistory" VALUES(3,'1122334455667',3,2018);
INSERT INTO "BookPublishingHistory" VALUES(4,'665544332211',4,2019);
INSERT INTO "BookPublishingHistory" VALUES(5,'778899110000',5,2017);
CREATE TABLE Books(
    ISBN VARCHAR(13) UNIQUE PRIMARY KEY,
    Title TEXT,
    YearOfPublication INT ,
    Genre TEXT,
    AuthorID INTEGER,
    FOREIGN KEY(AuthorID) REFERENCES Authors(AuthorID)
);
INSERT INTO "Books" VALUES('1234567890123','Science Fiction Book',2018,'Science Fiction',1);
INSERT INTO "Books" VALUES('9876543210987','Another Science Fiction Book',2020,'Science Fiction',1);
INSERT INTO "Books" VALUES('1122334455667','Romance Book',2018,'Romance',2);
INSERT INTO "Books" VALUES('665544332211','Yet Another Romance Book',2019,'Romance',2);
INSERT INTO "Books" VALUES('778899110000','Mystery Book',2017,'Mystery',3);
CREATE TABLE Publishers(
    PublisherID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT
);
INSERT INTO "Publishers" VALUES(1,'Publisher A','New York');
INSERT INTO "Publishers" VALUES(2,'Publisher B','London');
INSERT INTO "Publishers" VALUES(3,'Publisher C','Toronto');
INSERT INTO "Publishers" VALUES(4,'Publisher D','Sydney');
INSERT INTO "Publishers" VALUES(5,'Publisher E','Berlin');
CREATE TABLE Reviews(
    ReviewID INTEGER PRIMARY KEY AUTOINCREMENT,
    ISBN VARCHAR(13),
    Rating INT ,
    Comment TEXT,
    DatePosted DATE,
    FOREIGN KEY(ISBN) REFERENCES Books(ISBN)
);
INSERT INTO "Reviews" VALUES(1,'1234567890123',5,'Excellent book!','2021-08-01');
INSERT INTO "Reviews" VALUES(2,'9876543210987',4,'Good read.','2021-08-02');
INSERT INTO "Reviews" VALUES(3,'1122334455667',3,'Average story.','2021-08-03');
INSERT INTO "Reviews" VALUES(4,'665544332211',2,'Not my cup of tea.','2021-08-04');
INSERT INTO "Reviews" VALUES(5,'778899110000',1,'Did not like it at all.','2021-08-05');
CREATE TABLE SalesData(
    SaleID INTEGER PRIMARY KEY AUTOINCREMENT,
    ISBN VARCHAR(13),
    QuantitySold INT,
    Revenue DECIMAL(10,2),
    DateOfSale DATE,
    FOREIGN KEY(ISBN) REFERENCES Books(ISBN)
);
INSERT INTO "SalesData" VALUES(1,'1234567890123',500,10000,'2021-08-01');
INSERT INTO "SalesData" VALUES(2,'9876543210987',600,12000,'2021-08-02');
INSERT INTO "SalesData" VALUES(3,'1122334455667',400,8000,'2021-08-03');
INSERT INTO "SalesData" VALUES(4,'665544332211',300,6000,'2021-08-04');
INSERT INTO "SalesData" VALUES(5,'778899110000',200,4000,'2021-08-05');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Publishers',5);
INSERT INTO "sqlite_sequence" VALUES('BookPublishingHistory',5);
INSERT INTO "sqlite_sequence" VALUES('SalesData',5);
INSERT INTO "sqlite_sequence" VALUES('Reviews',5);
COMMIT;

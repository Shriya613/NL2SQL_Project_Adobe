BEGIN TRANSACTION;
CREATE TABLE ArticleSources(
    article_id INTEGER,
    source_id INTEGER,
    relationship TEXT,
    FOREIGN KEY (article_id) REFERENCES Articles(id),
    FOREIGN KEY (source_id) REFERENCES Sources(id));
INSERT INTO "ArticleSources" VALUES(1,1,'Primary Source');
INSERT INTO "ArticleSources" VALUES(1,2,'Secondary Source');
INSERT INTO "ArticleSources" VALUES(2,3,'Expert Interview');
INSERT INTO "ArticleSources" VALUES(2,4,'Background Information');
INSERT INTO "ArticleSources" VALUES(3,5,'Press Release');
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    deadline DATE,
    reporter_id INTEGER,
    photographer_id INTEGER,
    FOREIGN KEY (reporter_id) REFERENCES Reporters(id),
    FOREIGN KEY (photographer_id) REFERENCES Photographers(id));
INSERT INTO "Articles" VALUES(1,'Article Title 1','Content of Article 1','2020-04-25',1,1);
INSERT INTO "Articles" VALUES(2,'Article Title 2','Content of Article 2','2020-04-26',2,2);
INSERT INTO "Articles" VALUES(3,'Article Title 3','Content of Article 3','2020-04-27',3,3);
INSERT INTO "Articles" VALUES(4,'Article Title 4','Content of Article 4','2020-04-28',4,4);
INSERT INTO "Articles" VALUES(5,'Article Title 5','Content of Article 5','2020-04-29',5,5);
CREATE TABLE Photographers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    phone_number TEXT,
    address TEXT);
INSERT INTO "Photographers" VALUES(1,'David White','davidwhite@example.com','444-444-4444','234 Birch St');
INSERT INTO "Photographers" VALUES(2,'Eva Black','evablack@example.com','666-666-6666','345 Cedar St');
INSERT INTO "Photographers" VALUES(3,'Frank Red','frankred@example.com','777-777-7777','456 Walnut St');
INSERT INTO "Photographers" VALUES(4,'Grace Blue','graceblue@example.com','888-888-8888','567 Willow St');
INSERT INTO "Photographers" VALUES(5,'Hannah Green','hannahgreen@example.com','999-999-9999','678 Sycamore St');
CREATE TABLE ReaderComments(
    comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    reader_name TEXT,
    email TEXT,
    comment TEXT,
    article_id INTEGER,
    FOREIGN KEY (article_id) REFERENCES Articles(id));
INSERT INTO "ReaderComments" VALUES(1,'Reader 1','reader1@example.com','Great article!',1);
INSERT INTO "ReaderComments" VALUES(2,'Reader 2','reader2@example.com','Interesting perspective.',2);
INSERT INTO "ReaderComments" VALUES(3,'Reader 3','reader3@example.com','I disagree with the conclusion.',3);
INSERT INTO "ReaderComments" VALUES(4,'Reader 4','reader4@example.com','This is misleading information.',4);
INSERT INTO "ReaderComments" VALUES(5,'Reader 5','reader5@example.com','Well-written and informative.',5);
CREATE TABLE Reporters(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    phone_number TEXT,
    address TEXT);
INSERT INTO "Reporters" VALUES(1,'John Doe','johndoe@example.com','123-456-7890','123 Main St');
INSERT INTO "Reporters" VALUES(2,'Jane Smith','janesmith@example.com','098-765-4321','456 Elm St');
INSERT INTO "Reporters" VALUES(3,'Alice Johnson','alicejohnson@example.com','555-555-5555','789 Oak St');
INSERT INTO "Reporters" VALUES(4,'Bob Brown','bobbrown@example.com','333-333-3333','101 Pine St');
INSERT INTO "Reporters" VALUES(5,'Charlie Green','charliegreen@example.com','222-222-2222','111 Maple Ave');
CREATE TABLE Sources(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    contact_info TEXT,
    type TEXT );
INSERT INTO "Sources" VALUES(1,'Source A','sourcea@example.com','Government Agency');
INSERT INTO "Sources" VALUES(2,'Source B','sourceb@example.com','NGO');
INSERT INTO "Sources" VALUES(3,'Source C','sourcec@example.com','Private Company');
INSERT INTO "Sources" VALUES(4,'Source D','sourced@example.com','Academic Institution');
INSERT INTO "Sources" VALUES(5,'Source E','sourcee@example.com','Journalist Association');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Reporters',5);
INSERT INTO "sqlite_sequence" VALUES('Photographers',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Sources',5);
INSERT INTO "sqlite_sequence" VALUES('ReaderComments',5);
COMMIT;

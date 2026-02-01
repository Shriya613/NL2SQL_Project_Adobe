BEGIN TRANSACTION;
CREATE TABLE Collections (
    collection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_name TEXT
);
INSERT INTO "Collections" VALUES(1,'Children''s Books');
INSERT INTO "Collections" VALUES(2,'Young Adult');
INSERT INTO "Collections" VALUES(3,'Non-Fiction');
INSERT INTO "Collections" VALUES(4,'Biography');
INSERT INTO "Collections" VALUES(5,'History');
CREATE TABLE Libraries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    phone_number TEXT
);
INSERT INTO "Libraries" VALUES(1,'Library A','123 Main St.','555-1234');
INSERT INTO "Libraries" VALUES(2,'Library B','456 Elm St.','555-5678');
INSERT INTO "Libraries" VALUES(3,'Library C','789 Oak St.','555-9012');
INSERT INTO "Libraries" VALUES(4,'Library D','321 Pine St.','555-1357');
INSERT INTO "Libraries" VALUES(5,'Library E','654 Fir St.','555-1415');
CREATE TABLE Libraries_Collections (
    library_id INTEGER,
    collection_id INTEGER,
    FOREIGN KEY(library_id) REFERENCES Libraries(id),
    FOREIGN KEY(collection_id) REFERENCES Collections(collection_id)
);
INSERT INTO "Libraries_Collections" VALUES(1,1);
INSERT INTO "Libraries_Collections" VALUES(2,1);
INSERT INTO "Libraries_Collections" VALUES(3,1);
INSERT INTO "Libraries_Collections" VALUES(4,1);
INSERT INTO "Libraries_Collections" VALUES(5,1);
INSERT INTO "Libraries_Collections" VALUES(5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Libraries',5);
INSERT INTO "sqlite_sequence" VALUES('Collections',5);
COMMIT;

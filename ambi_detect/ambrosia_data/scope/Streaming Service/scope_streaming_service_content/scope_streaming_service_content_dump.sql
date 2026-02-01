BEGIN TRANSACTION;
CREATE TABLE Content (
    content_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    description TEXT,
    price REAL 
);
INSERT INTO "Content" VALUES(1,'Movie','A thrilling action movie.',3.99);
INSERT INTO "Content" VALUES(2,'Series','An exciting crime series.',2.99);
INSERT INTO "Content" VALUES(3,'Documentary','An educational documentary on space exploration.',0.99);
INSERT INTO "Content" VALUES(4,'Kids Show','A fun-filled show for kids.',1.99);
INSERT INTO "Content" VALUES(5,'Animation','An engaging animated film.',2.99);
INSERT INTO "Content" VALUES(6,'Netflix Originals','Exclusive content available only on Netflix.',4.99);
CREATE TABLE Streaming_Service (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    subscription_fee REAL 
);
INSERT INTO "Streaming_Service" VALUES(1,'Netflix',7.99);
INSERT INTO "Streaming_Service" VALUES(2,'Hulu',5.99);
INSERT INTO "Streaming_Service" VALUES(3,'Amazon Prime Video',8.99);
INSERT INTO "Streaming_Service" VALUES(4,'Disney+',6.99);
INSERT INTO "Streaming_Service" VALUES(5,'HBO Max',14.99);
CREATE TABLE Streaming_Service_Content (
    streaming_service_id INTEGER,
    content_id INTEGER,
    FOREIGN KEY (streaming_service_id) REFERENCES Streaming_Service(id),
    FOREIGN KEY (content_id) REFERENCES Content(content_id),
    PRIMARY KEY (streaming_service_id, content_id)
);
INSERT INTO "Streaming_Service_Content" VALUES(1,1);
INSERT INTO "Streaming_Service_Content" VALUES(1,6);
INSERT INTO "Streaming_Service_Content" VALUES(2,2);
INSERT INTO "Streaming_Service_Content" VALUES(2,5);
INSERT INTO "Streaming_Service_Content" VALUES(3,3);
INSERT INTO "Streaming_Service_Content" VALUES(3,4);
INSERT INTO "Streaming_Service_Content" VALUES(4,1);
INSERT INTO "Streaming_Service_Content" VALUES(4,6);
INSERT INTO "Streaming_Service_Content" VALUES(5,2);
INSERT INTO "Streaming_Service_Content" VALUES(5,5);
INSERT INTO "Streaming_Service_Content" VALUES(2,1);
INSERT INTO "Streaming_Service_Content" VALUES(3,1);
INSERT INTO "Streaming_Service_Content" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streaming_Service',5);
INSERT INTO "sqlite_sequence" VALUES('Content',6);
COMMIT;

#include <assert.h>
#include "leveldb/db.h"
//using namespace leveldb;
int main()
{

leveldb::DB* db;
leveldb::Options options;

leveldb::Options options;
options.create_if_missing = true;
leveldb::Status status = leveldb::DB::Open(options, "~/caffedeep/caffe-master/examples/_temp/feat_conv1", &db);
assert(status.ok());
}

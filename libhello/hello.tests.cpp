#include <gtest/gtest.h>
#include <libhello/hello.hpp>

TEST(helloClass, helloStuff_GivingOne_ReturnsOne)
{
    Hello a;
    EXPECT_EQ(a.helloStuff(1),1);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

package category;

import java.util.*;

public class PDD {
    public static void main(String[] args) {
        PDD pDD = new PDD();
        pDD.numbers();
    }


    //    面经的算法题
//    用1---9九个数码组成三个三位数,要求第二个数,第三个数分别是第一个数的2倍和3倍,一共有几组?
//    就只用九个数字组成三个三位数,不能重复使用数字
    public void numbers() {
        int[] nums = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        List<List<Integer>> res = new ArrayList<>();
        allNums(nums, 0, new ArrayDeque<>(), res);
        System.out.println(res);
    }

    public void allNums(int[] nums, int i, Deque<Integer> q, List<List<Integer>> res) {

//        if (i >= nums.length) {
//            res.add(new ArrayList<>(q));
//            return;
//        }
//        剪枝
        if (q.size() == 6) {
//            456
            int one = 0, two = 0;
            List<Integer> t = new ArrayList<>(q);
            for (int i1 = t.size() - 1; i1 >= 0; i1--) {
                if (i1 < 3) {
                    one *= 10;
                    one += t.get(i1);
                } else {
                    two *= 10;
                    two += t.get(i1);
                }
            }
            if (two / one != 2 || two % one != 0) {
                return;
            } else {
                System.out.println(one + "*2=" + two);
            }
        }
        //        剪枝
        if (q.size() == 9) {
//            456
            int one = 0, two = 0, three = 0;
            List<Integer> t = new ArrayList<>(q);
            for (int i1 = t.size() - 1; i1 >= 0; i1--) {
                if (i1 < 3) {
                    one *= 10;
                    one += t.get(i1);
                } else if (i1 >= 3 && i1 < 6) {
                    two *= 10;
                    two += t.get(i1);
                } else {
                    three *= 10;
                    three += t.get(i1);
                }
            }
            if (three / one == 3 && three % one == 0 && two / one == 2 && two % one == 0) {
                System.out.println(one + "*2=" + two + "," + one + "*3=" + three);
                Collections.reverse(t);
                res.add(t);
            } else {
                return;
            }
        }

        for (int j = i; j < nums.length; j++) {
            MyUtile.swap(nums, i, j);
            q.offerLast(nums[i]);
            allNums(nums, i + 1, q, res);
            MyUtile.swap(nums, i, j);
            q.pollLast();
        }
    }
//    烧饼排序
}

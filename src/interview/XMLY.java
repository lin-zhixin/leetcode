package interview;

import javafx.util.Pair;

import javax.swing.tree.TreeNode;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

//class TreeNode {
//    int val;
//    TreeNode left;
//    TreeNode right;
//
//    TreeNode() {
//    }
//
//    TreeNode(int val) {
//        this.val = val;
//    }
//
//    TreeNode(int val, TreeNode left, TreeNode right) {
//        this.val = val;
//        this.left = left;
//        this.right = right;
//    }
//}

public class XMLY {
    public Pair<Integer, Integer> lower_upper(int[] nums, int target) {
        int l = lower_bound(nums, target);
        int r = upper_bound(nums, target);
        if (nums[l] != target) l = -1;
        if (r > nums.length - 1 || r == 0) {
            r = -1;
        } else {
            r--;
        }
        return new Pair<>(l, r);
    }

    public int lower_bound(int[] nums, int target) {
        int n = nums.length, l = 0, r = n - 1;
        while (l <= r) {
//            []
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                r = mid - 1;
            } else if (nums[mid] < target) {
                l = mid + 1;
            } else if (nums[mid] > target) {
                r = mid - 1;
            }
        }
        return l;
    }

    public int upper_bound(int[] nums, int target) {
        int n = nums.length, l = 0, r = n - 1;
        while (l <= r) {
//            []
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                l = mid + 1;
            } else if (nums[mid] < target) {
                l = mid + 1;
            } else if (nums[mid] > target) {
                r = mid - 1;
            }
        }
        return l;
    }


//    public List<Integer> postorderTraversal(TreeNode root) {
//        Stack<TreeNode> stack = new Stack<TreeNode>();
//        List<Integer> res = new ArrayList<>();
//        TreeNode cur = null, pre = null;
//        while (!stack.isEmpty() || root != null) {
//            while (root != null) {
//                stack.push(root);
//                root = root.left;
//            }
//            cur = stack.peek();
//            if (cur.right != null && cur.right != pre) {
//                root = cur.right;
//            } else {
//                pre = stack.pop();
//                res.add(pre.val);
//            }
//
//        }
//        return res;
//
//
//    }

    public static void main(String[] args) {
        XMLY xmly = new XMLY();
        System.out.println(xmly.lower_upper(new int[]{1, 2, 3, 3, 3, 3, 4, 5, 6}, 7));
    }
}

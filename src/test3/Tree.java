package test3;


import java.util.*;
import java.util.stream.Collectors;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

public class Tree {
    //1325. 删除给定值的叶子节点
    public TreeNode removeLeafNodes(TreeNode root, int target) {
        if (root == null) {
            return null;
        }
        if (root.left == null && root.right == null && root.val == target) {
            return root = null;
        }
        root.left = removeLeafNodes(root.left, target);
        root.right = removeLeafNodes(root.right, target);
        if (root.left == null && root.right == null && root.val == target) {
            return root = null;
        }
        return root;
    }

    //    662. 二叉树最大宽度
    public int widthOfBinaryTree(TreeNode root) {
        int max = 0;
        Deque<TreeNode> q = new LinkedList<>();
        Map<TreeNode, Integer> map = new HashMap<>();
        q.offerLast(root);
        map.put(root, 1);
        while (!q.isEmpty()) {
            max = Math.max(max, map.get(q.peekLast()) - map.get(q.peekFirst()) + 1);
            int levelSize = q.size();
//            System.out.println(levelSize);
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = q.pollFirst();
                if (node.left != null) {
                    q.offerLast(node.left);
                    map.put(node.left, map.get(node) * 2);
                }
                if (node.right != null) {
                    q.offerLast(node.right);
                    map.put(node.right, map.get(node) * 2 + 1);
                }
            }

        }
        return max;
    }

    //103. 二叉树的锯齿形层序遍历
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        boolean re = false;
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Deque<TreeNode> q = new LinkedList<>();
        q.offerLast(root);
        while (!q.isEmpty()) {
            List<Integer> t = q.stream().map(e -> e.val).collect(Collectors.toList());
            if (re) {
                Collections.reverse(t);
            }
            res.add(t);
            re = !re;
            int levelSize = q.size();
//            System.out.println(levelSize);
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = q.pollFirst();
                if (node.left != null) {
                    q.offerLast(node.left);
                }
                if (node.right != null) {
                    q.offerLast(node.right);
                }
            }
        }
        return res;
    }
}
